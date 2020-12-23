package tabnet

import (
	"fmt"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// TabNetOpts contains parameters to configure the tab net algorithm
type TabNetOpts struct {
	OutputFeatures    int
	SharedBlocks      int
	IndependentBlocks int
	DecisionSteps     int

	InputDim           int // FIXME
	PredictionLayerDim int
	AttentionLayerDim  int

	Gamma float64

	Momentum                         float64
	Epsilon                          float64
	VirtualBatchSize                 int
	Inferring                        bool
	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

// TabNet implements the tab net architecture
func (nn *Model) TabNet(opts TabNetOpts) Layer {
	shared := make([]Layer, 0, opts.SharedBlocks)

	for i := 0; i < opts.SharedBlocks; i++ {
		shared = append(shared, nn.FC(FCOpts{
			OutputFeatures: 2 * (opts.PredictionLayerDim + opts.AttentionLayerDim), // double the size so we can take half and half
			WeightsInit:    opts.WeightsInit,
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
				OutputFeatures:     opts.InputDim,
			},
		))
	}

	fcLayer := nn.FC(FCOpts{
		OutputFeatures: opts.OutputFeatures,
		WeightsInit:    opts.WeightsInit,
	})

	bnLayer := nn.BN(BNOpts{ // TODO: make configurable
		ScaleInit: opts.ScaleInit,
		BiasInit:  opts.BiasInit,
		Inferring: opts.Inferring,
		InputSize: opts.OutputFeatures,
	})

	// first step
	ftLayer := nn.FeatureTransformer(FeatureTransformerOpts{
		Shared:            shared,
		VirtualBatchSize:  opts.VirtualBatchSize,
		IndependentBlocks: opts.IndependentBlocks,
		OutputFeatures:    opts.AttentionLayerDim + opts.PredictionLayerDim,
		WeightsInit:       opts.WeightsInit,
	})

	if opts.Gamma == 0.0 {
		opts.Gamma = 1.2
	}

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
