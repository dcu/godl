package tabnet

import (
	"fmt"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// TabNetOpts contains parameters to configure the tab net algorithm
type TabNetOpts struct {
	OutputDimension   int
	SharedBlocks      int
	IndependentBlocks int
	DecisionSteps     int

	InputDim  int // FIXME
	BatchSize int

	PredictionLayerDim int
	AttentionLayerDim  int

	MaskFunction ActivationFn

	Gamma float64

	Momentum                         float64
	Epsilon                          float64
	VirtualBatchSize                 int
	Inferring                        bool
	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

func (o *TabNetOpts) setDefaults() {
	if o.SharedBlocks == 0 {
		o.SharedBlocks = 2
	}

	if o.IndependentBlocks == 0 {
		o.SharedBlocks = 2
	}

	if o.DecisionSteps == 0 {
		o.DecisionSteps = 3
	}

	if o.PredictionLayerDim == 0 {
		o.PredictionLayerDim = 8
	}

	if o.AttentionLayerDim == 0 {
		o.PredictionLayerDim = 8
	}

	if o.Gamma == 0.0 {
		o.Gamma = 1.3
	}
}

// TabNet implements the tab net architecture
func (nn *Model) TabNet(opts TabNetOpts) Layer {
	opts.setDefaults()

	shared := make([]Layer, 0, opts.SharedBlocks)

	for i := 0; i < opts.SharedBlocks; i++ {
		shared = append(shared, nn.FC(FCOpts{
			InputDimension:  opts.BatchSize,
			OutputDimension: 2 * (opts.PredictionLayerDim + opts.AttentionLayerDim), // double the size so we can take half and half
			WeightsInit:     opts.WeightsInit,
			WithBias:        true,
		}))
	}

	steps := make([]*DecisionStep, 0, opts.DecisionSteps)
	for i := 0; i < opts.DecisionSteps-1; i++ {
		steps = append(steps, nn.DecisionStep(
			DecisionStepOpts{
				Shared:             shared,
				IndependentBlocks:  opts.IndependentBlocks,
				VirtualBatchSize:   opts.VirtualBatchSize,
				PredictionLayerDim: opts.PredictionLayerDim,
				AttentionLayerDim:  opts.AttentionLayerDim,
				WeightsInit:        opts.WeightsInit,
				ScaleInit:          opts.ScaleInit,
				BiasInit:           opts.BiasInit,
				Inferring:          opts.Inferring,
				InputDimension:     opts.BatchSize,
				OutputDimension:    opts.InputDim,
				MaskFunction:       opts.MaskFunction,
			},
		))
	}

	fcLayer := nn.FC(FCOpts{
		InputDimension:  opts.BatchSize,
		OutputDimension: opts.OutputDimension,
		WeightsInit:     opts.WeightsInit,
		WithBias:        true,
	})

	bnLayer := nn.BN(BNOpts{ // TODO: make configurable
		ScaleInit: opts.ScaleInit,
		BiasInit:  opts.BiasInit,
		Inferring: opts.Inferring,
		InputSize: opts.OutputDimension,
	})

	// first step
	ftLayer := nn.FeatureTransformer(FeatureTransformerOpts{
		Shared:            shared,
		VirtualBatchSize:  opts.VirtualBatchSize,
		IndependentBlocks: opts.IndependentBlocks,
		OutputDimension:   opts.AttentionLayerDim + opts.PredictionLayerDim,
		WeightsInit:       opts.WeightsInit,
	})

	gamma := gorgonia.NewScalar(nn.g, tensor.Float64, gorgonia.WithValue(opts.Gamma))

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, error) {
		x := nodes[0]
		xShape := x.Shape()

		bn, err := bnLayer(x)
		if err != nil {
			return nil, err
		}

		ft, err := ftLayer(bn)
		if err != nil {
			return nil, err
		}

		xAttentiveLayer, err := gorgonia.Slice(ft, nil, gorgonia.S(opts.PredictionLayerDim, ft.Shape()[1]))
		if err != nil {
			return nil, err
		}

		prior := gorgonia.NewTensor(nn.g, tensor.Float64, bn.Shape().Dims(), gorgonia.WithShape(bn.Shape()...), gorgonia.WithInit(gorgonia.Ones()), gorgonia.WithName("Prior"))
		out := gorgonia.NewTensor(nn.g, tensor.Float64, 2, gorgonia.WithShape(xShape[0], opts.PredictionLayerDim), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName("Output"))

		for _, step := range steps {
			mask, err := step.AttentiveTransformer(xAttentiveLayer, prior)
			if err != nil {
				return nil, fmt.Errorf("attentive transformer: %w", err)
			}

			// Update prior
			{
				prior, err = gorgonia.HadamardProd(prior, gorgonia.Must(gorgonia.Sub(gamma, mask)))
				if err != nil {
					return nil, fmt.Errorf("updating prior: %w", err)
				}
			}

			ds, err := step.FeatureTransformer(bn, mask)
			if err != nil {
				return nil, err
			}

			firstPart, err := gorgonia.Slice(ds, nil, gorgonia.S(0, opts.PredictionLayerDim))
			if err != nil {
				return nil, err
			}

			relu, err := gorgonia.Rectify(firstPart)
			if err != nil {
				return nil, err
			}

			out, err = gorgonia.Add(out, relu)
			if err != nil {
				return nil, err
			}

			xAttentiveLayer, err = gorgonia.Slice(ds, nil, gorgonia.S(opts.PredictionLayerDim, ds.Shape()[1]))
			if err != nil {
				return nil, err
			}
		}

		return fcLayer(out)
	}
}
