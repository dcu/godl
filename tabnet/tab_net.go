package tabnet

import (
	"fmt"
	"math"

	"github.com/dcu/deepzen"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// TabNetOpts contains parameters to configure the tab net algorithm
type TabNetOpts struct {
	OutputSize int
	InputSize  int
	BatchSize  int

	SharedBlocks       int
	IndependentBlocks  int
	DecisionSteps      int
	PredictionLayerDim int
	AttentionLayerDim  int

	MaskFunction deepzen.ActivationFn

	WithBias bool

	Gamma                            float32
	Momentum                         float32
	Epsilon                          float32
	VirtualBatchSize                 int
	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

func (o *TabNetOpts) setDefaults() {
	if o.SharedBlocks == 0 {
		o.SharedBlocks = 2
	}

	if o.IndependentBlocks == 0 {
		o.IndependentBlocks = 2
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
		o.Epsilon = 1e-15
	}

	if o.Gamma == 0.0 {
		o.Gamma = 1.3
	}

	if o.Momentum == 0 {
		o.Momentum = 0.02
	}

	if o.VirtualBatchSize == 0 {
		o.VirtualBatchSize = 128
	}
}

// TabNet implements the tab net architecture
func TabNet(nn *deepzen.Model, opts TabNetOpts) deepzen.Layer {
	opts.setDefaults()

	bnLayer := deepzen.BatchNorm1d(nn, deepzen.BatchNormOpts{
		ScaleInit: opts.ScaleInit,
		BiasInit:  opts.BiasInit,
		InputSize: opts.InputSize,
		Momentum:  0.01,
	})

	shared := make([]deepzen.Layer, 0, opts.SharedBlocks)
	outputDim := 2 * (opts.PredictionLayerDim + opts.AttentionLayerDim) // double the size so we can take half and half

	{
		fcInput := opts.InputSize
		fcOutput := outputDim

		for i := 0; i < opts.SharedBlocks; i++ {
			sharedWeightsInit := opts.WeightsInit

			if sharedWeightsInit == nil {
				gain := math.Sqrt(float64(fcInput+fcOutput*2) / math.Sqrt(float64(fcInput)))

				sharedWeightsInit = gorgonia.GlorotN(gain)
			}

			shared = append(shared, deepzen.FC(nn, deepzen.FCOpts{
				InputDimension:  fcInput,
				OutputDimension: fcOutput,
				WeightsInit:     sharedWeightsInit,
				WithBias:        opts.WithBias,
			}))

			fcInput = opts.PredictionLayerDim + opts.AttentionLayerDim
		}
	}

	// first step
	initialSplitter := FeatureTransformer(nn, FeatureTransformerOpts{
		Shared:            shared,
		VirtualBatchSize:  opts.VirtualBatchSize,
		IndependentBlocks: opts.IndependentBlocks,
		InputDimension:    opts.InputSize,
		OutputDimension:   opts.AttentionLayerDim + opts.PredictionLayerDim,
		WeightsInit:       opts.WeightsInit,
		WithBias:          opts.WithBias,
		Momentum:          opts.Momentum,
	})

	steps := make([]*DecisionStep, 0, opts.DecisionSteps)
	for i := 0; i < opts.DecisionSteps; i++ {
		steps = append(steps, NewDecisionStep(nn,
			DecisionStepOpts{
				Shared:             shared,
				IndependentBlocks:  opts.IndependentBlocks,
				VirtualBatchSize:   opts.VirtualBatchSize,
				PredictionLayerDim: opts.PredictionLayerDim,
				AttentionLayerDim:  opts.AttentionLayerDim,
				WeightsInit:        opts.WeightsInit,
				ScaleInit:          opts.ScaleInit,
				BiasInit:           opts.BiasInit,
				InputDimension:     opts.BatchSize,
				OutputDimension:    opts.InputSize,
				MaskFunction:       opts.MaskFunction,
				WithBias:           opts.WithBias,
				Momentum:           opts.Momentum,
			},
		))
	}

	weightsInit := opts.WeightsInit
	if weightsInit == nil {
		gain := math.Sqrt(float64(opts.PredictionLayerDim+opts.OutputSize) / math.Sqrt(float64(4*opts.PredictionLayerDim)))
		weightsInit = gorgonia.GlorotN(gain)
	}

	finalMapping := deepzen.FC(nn, deepzen.FCOpts{
		InputDimension:  opts.PredictionLayerDim,
		OutputDimension: opts.OutputSize,
		WeightsInit:     weightsInit,
		WithBias:        opts.WithBias,
	})

	gamma := gorgonia.NewConstant(opts.Gamma)
	epsilon := gorgonia.NewConstant(opts.Epsilon)

	tabNetLoss := gorgonia.NewScalar(nn.ExprGraph(), tensor.Float32, gorgonia.WithValue(float32(0.0)), gorgonia.WithName("TabNetLoss"))
	stepsCount := gorgonia.NewScalar(nn.ExprGraph(), tensor.Float32, gorgonia.WithValue(float32(len(steps))), gorgonia.WithName("Steps"))

	prior := gorgonia.NewTensor(nn.ExprGraph(), tensor.Float32, 2, gorgonia.WithShape(opts.BatchSize, opts.InputSize), gorgonia.WithInit(gorgonia.Ones()), gorgonia.WithName("Prior"))
	out := gorgonia.NewTensor(nn.ExprGraph(), tensor.Float32, 2, gorgonia.WithShape(opts.BatchSize, opts.PredictionLayerDim), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName("Output"))

	return func(nodes ...*gorgonia.Node) (deepzen.Result, error) {
		x := nodes[0]

		bn, err := bnLayer(x)
		if err != nil {
			return deepzen.Result{}, fmt.Errorf("applying initial batch norm %v: %w", x.Shape(), err)
		}

		ft, err := initialSplitter(bn.Output)
		if err != nil {
			return deepzen.Result{}, fmt.Errorf("applying initial splitter %v: %w", bn.Output.Shape(), err)
		}

		xAttentiveLayer, err := gorgonia.Slice(ft.Output, nil, gorgonia.S(opts.PredictionLayerDim, ft.Shape()[1]))
		if err != nil {
			return deepzen.Result{}, fmt.Errorf("slicing %v: %w", ft.Shape(), err)
		}

		for _, step := range steps {
			mask, stepLoss, err := step.CalculateMask(xAttentiveLayer, prior, epsilon)
			if err != nil {
				return deepzen.Result{}, err
			}

			// accum losses
			tabNetLoss = gorgonia.Must(gorgonia.Add(tabNetLoss, stepLoss))

			// Update prior
			{
				prior, err = gorgonia.HadamardProd(gorgonia.Must(gorgonia.Sub(gamma, mask)), prior)
				if err != nil {
					return deepzen.Result{}, fmt.Errorf("updating prior: %w", err)
				}
			}

			maskedX, err := gorgonia.HadamardProd(mask, bn.Output)
			if err != nil {
				return deepzen.Result{}, err
			}

			ds, err := step.FeatureTransformer(maskedX)
			if err != nil {
				return deepzen.Result{}, err
			}

			firstPart, err := gorgonia.Slice(ds.Output, nil, gorgonia.S(0, opts.PredictionLayerDim))
			if err != nil {
				return deepzen.Result{}, err
			}

			relu, err := gorgonia.Rectify(firstPart)
			if err != nil {
				return deepzen.Result{}, err
			}

			out, err = gorgonia.Add(out, relu)
			if err != nil {
				return deepzen.Result{}, err
			}

			xAttentiveLayer, err = gorgonia.Slice(ds.Output, nil, gorgonia.S(opts.PredictionLayerDim, ds.Shape()[1]))
			if err != nil {
				return deepzen.Result{}, err
			}
		}

		tabNetLoss = gorgonia.Must(gorgonia.Div(tabNetLoss, stepsCount))

		result, err := finalMapping(out)
		if err != nil {
			return deepzen.Result{}, fmt.Errorf("TabNet: applying final FC layer to %v: %w", out.Shape(), err)
		}

		result.Loss = tabNetLoss

		return result, nil
	}
}
