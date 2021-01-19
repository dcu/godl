package tabnet

import (
	"fmt"
	"math"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// TabNetOpts contains parameters to configure the tab net algorithm
type TabNetOpts struct {
	OutputDimension   int
	SharedBlocks      int
	IndependentBlocks int
	DecisionSteps     int

	InputDimension int // FIXME
	BatchSize      int

	PredictionLayerDim int
	AttentionLayerDim  int

	MaskFunction ActivationFn

	Gamma float64

	WithBias bool

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
		o.AttentionLayerDim = 8
	}

	if o.Epsilon == 0.0 {
		o.Epsilon = 1e-5
	}

	if o.Gamma == 0.0 {
		o.Gamma = 1.3
	}

	if o.VirtualBatchSize == 0 {
		o.VirtualBatchSize = 128
	}
}

// TabNet implements the tab net architecture
func (nn *Model) TabNet(opts TabNetOpts) Layer {
	opts.setDefaults()

	bnLayer := nn.BN(BNOpts{ // TODO: make configurable
		ScaleInit: opts.ScaleInit,
		BiasInit:  opts.BiasInit,
		Inferring: opts.Inferring,
		InputDim:  opts.BatchSize,
		OutputDim: opts.InputDimension,
	})

	shared := make([]Layer, 0, opts.SharedBlocks)
	outputDim := 2 * (opts.PredictionLayerDim + opts.AttentionLayerDim) // double the size so we can take half and half

	{
		fcInput := opts.InputDimension
		fcOutput := outputDim
		sharedWeightsInit := opts.WeightsInit

		if sharedWeightsInit == nil {
			gain := math.Sqrt(float64(fcInput+fcOutput) / math.Sqrt(float64(fcInput)))
			sharedWeightsInit = gorgonia.GlorotN(gain)
		}

		for i := 0; i < opts.SharedBlocks; i++ {
			shared = append(shared, nn.FC(FCOpts{
				InputDimension:  fcInput,
				OutputDimension: fcOutput,
				WeightsInit:     sharedWeightsInit,
				WithBias:        opts.WithBias,
			}))

			fcInput = opts.PredictionLayerDim + opts.AttentionLayerDim
		}
	}

	// first step
	initialSplitter := nn.FeatureTransformer(FeatureTransformerOpts{
		Shared:            shared,
		VirtualBatchSize:  opts.VirtualBatchSize,
		IndependentBlocks: opts.IndependentBlocks,
		InputDimension:    opts.InputDimension,
		OutputDimension:   opts.AttentionLayerDim + opts.PredictionLayerDim,
		WeightsInit:       opts.WeightsInit,
		WithBias:          opts.WithBias,
	})

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
				OutputDimension:    opts.InputDimension,
				MaskFunction:       opts.MaskFunction,
				WithBias:           opts.WithBias,
			},
		))
	}

	weightsInit := opts.WeightsInit
	if weightsInit == nil {
		gain := math.Sqrt(float64(opts.PredictionLayerDim+opts.OutputDimension) / math.Sqrt(float64(4*opts.PredictionLayerDim)))
		weightsInit = gorgonia.GlorotN(gain)
	}

	finalMapping := nn.FC(FCOpts{
		InputDimension:  opts.PredictionLayerDim,
		OutputDimension: opts.OutputDimension,
		WeightsInit:     weightsInit,
		WithBias:        opts.WithBias,
	})

	gamma := gorgonia.NewConstant(opts.Gamma)
	epsilon := gorgonia.NewConstant(opts.Epsilon)

	tabNetLoss := gorgonia.NewScalar(nn.g, tensor.Float64, gorgonia.WithValue(0.0), gorgonia.WithName("TabNetLoss"))
	stepsCount := gorgonia.NewScalar(nn.g, tensor.Float64, gorgonia.WithValue(float64(len(steps))), gorgonia.WithName("Steps"))

	prior := gorgonia.NewTensor(nn.g, tensor.Float64, 2, gorgonia.WithShape(opts.BatchSize, opts.InputDimension), gorgonia.WithInit(gorgonia.Ones()), gorgonia.WithName("Prior"))
	out := gorgonia.NewTensor(nn.g, tensor.Float64, 2, gorgonia.WithShape(opts.BatchSize, opts.PredictionLayerDim), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName("Output"))

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, error) {
		x := nodes[0]

		bn, _, err := bnLayer(x)
		if err != nil {
			return nil, nil, fmt.Errorf("applying initial batch norm %v: %w", x.Shape(), err)
		}

		ft, _, err := initialSplitter(bn)
		if err != nil {
			return nil, nil, fmt.Errorf("applying initial splitter %v: %w", bn.Shape(), err)
		}

		xAttentiveLayer, err := gorgonia.Slice(ft, nil, gorgonia.S(opts.PredictionLayerDim, ft.Shape()[1]))
		if err != nil {
			return nil, nil, fmt.Errorf("slicing %v: %w", ft.Shape(), err)
		}

		for _, step := range steps {
			mask, _, err := step.AttentiveTransformer(xAttentiveLayer, prior)
			if err != nil {
				return nil, nil, fmt.Errorf("attentive transformer: %w", err)
			}

			// Update prior
			{
				prior, err = gorgonia.HadamardProd(prior, gorgonia.Must(gorgonia.Sub(gamma, mask)))
				if err != nil {
					return nil, nil, fmt.Errorf("updating prior: %w", err)
				}
			}

			lg := gorgonia.Must(gorgonia.Log(
				gorgonia.Must(gorgonia.Add(mask, epsilon)),
			))

			stepLoss := gorgonia.Must(gorgonia.Mean(
				gorgonia.Must(gorgonia.Sum(
					gorgonia.Must(gorgonia.HadamardProd(
						mask,
						lg,
					)),
					1,
				)),
			))

			tabNetLoss = gorgonia.Must(gorgonia.Add(tabNetLoss, stepLoss))

			ds, _, err := step.FeatureTransformer(bn, mask)
			if err != nil {
				return nil, nil, err
			}

			firstPart, err := gorgonia.Slice(ds, nil, gorgonia.S(0, opts.PredictionLayerDim))
			if err != nil {
				return nil, nil, err
			}

			relu, err := gorgonia.Rectify(firstPart)
			if err != nil {
				return nil, nil, err
			}

			out, err = gorgonia.Add(out, relu)
			if err != nil {
				return nil, nil, err
			}

			xAttentiveLayer, err = gorgonia.Slice(ds, nil, gorgonia.S(opts.PredictionLayerDim, ds.Shape()[1]))
			if err != nil {
				return nil, nil, err
			}
		}

		output, _, err := finalMapping(out)
		if err != nil {
			return nil, nil, fmt.Errorf("TabNet: applying final FC layer to %v: %w", out.Shape(), err)
		}

		tabNetLoss = gorgonia.Must(gorgonia.Div(tabNetLoss, stepsCount))

		return output, tabNetLoss, nil
	}
}
