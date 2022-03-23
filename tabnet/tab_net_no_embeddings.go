package tabnet

import (
	"fmt"
	"math"

	"github.com/dcu/godl"
	"github.com/dcu/godl/activation"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// TabNetNoEmbeddingsOpts contains parameters to configure the tab net algorithm
type TabNetNoEmbeddingsOpts struct {
	OutputSize int
	InputSize  int
	BatchSize  int

	SharedBlocks       int
	IndependentBlocks  int
	DecisionSteps      int
	PredictionLayerDim int
	AttentionLayerDim  int

	MaskFunction activation.Function

	WithBias bool

	Gamma                            float64
	Momentum                         float64
	Epsilon                          float64
	VirtualBatchSize                 int
	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

func (o *TabNetNoEmbeddingsOpts) setDefaults() {
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

// TabNetNoEmbeddings implements the tab net architecture
func TabNetNoEmbeddings(nn *godl.Model, opts TabNetNoEmbeddingsOpts) godl.Layer {
	opts.setDefaults()

	bnLayer := godl.BatchNorm1d(nn, godl.BatchNormOpts{
		ScaleInit: opts.ScaleInit,
		BiasInit:  opts.BiasInit,
		InputSize: opts.InputSize,
		Momentum:  0.01,
	})

	shared := make([]godl.Layer, 0, opts.SharedBlocks)
	outputDim := 2 * (opts.PredictionLayerDim + opts.AttentionLayerDim) // double the size so we can take half and half

	{
		fcInput := opts.InputSize
		fcOutput := outputDim

		for i := 0; i < opts.SharedBlocks; i++ {
			sharedWeightsInit := opts.WeightsInit

			if sharedWeightsInit == nil {
				gain := math.Sqrt(float64(fcInput+fcOutput) / math.Sqrt(float64(fcInput)))

				sharedWeightsInit = gorgonia.GlorotN(gain)
			}

			shared = append(shared, godl.FC(nn, godl.FCOpts{
				InputDimension:  fcInput,
				OutputDimension: fcOutput,
				WeightsInit:     sharedWeightsInit,
				WithBias:        opts.WithBias,
				WeightsName:     fmt.Sprintf("shared.weight.%d", i),
				BiasName:        fmt.Sprintf("shared.bias.%d", i),
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

	featureTransformers := make([]godl.Layer, 0, opts.DecisionSteps)
	attentiveTransformers := make([]godl.Layer, 0, opts.DecisionSteps)

	for i := 0; i < opts.DecisionSteps; i++ {
		featureTransformer := FeatureTransformer(nn, FeatureTransformerOpts{
			Shared:            shared,
			VirtualBatchSize:  opts.VirtualBatchSize,
			InputDimension:    opts.BatchSize,
			OutputDimension:   opts.AttentionLayerDim + opts.PredictionLayerDim,
			IndependentBlocks: opts.IndependentBlocks,
			WeightsInit:       opts.WeightsInit,
			WithBias:          opts.WithBias,
			Momentum:          opts.Momentum,
		})
		featureTransformers = append(featureTransformers, featureTransformer)
	}

	for i := 0; i < opts.DecisionSteps; i++ {
		attentiveTransformer := AttentiveTransformer(nn, AttentiveTransformerOpts{
			InputDimension:   opts.AttentionLayerDim, // or prediction?
			OutputDimension:  opts.InputSize,
			Momentum:         opts.Momentum,
			Epsilon:          opts.Epsilon,
			VirtualBatchSize: opts.VirtualBatchSize,
			ScaleInit:        opts.ScaleInit,
			BiasInit:         opts.BiasInit,
			WeightsInit:      opts.WeightsInit,
			Activation:       opts.MaskFunction,
			WithBias:         opts.WithBias,
		})
		attentiveTransformers = append(attentiveTransformers, attentiveTransformer)
	}

	weightsInit := opts.WeightsInit
	if weightsInit == nil {
		gain := math.Sqrt(float64(opts.PredictionLayerDim+opts.OutputSize) / math.Sqrt(float64(4*opts.PredictionLayerDim)))

		weightsInit = gorgonia.GlorotN(gain)
	}

	finalMapping := godl.FC(nn, godl.FCOpts{
		InputDimension:  opts.PredictionLayerDim,
		OutputDimension: opts.OutputSize,
		WeightsInit:     weightsInit,
		WeightsName:     "FinalMapping",
		WithBias:        opts.WithBias,
	})

	gamma := gorgonia.NewConstant(float32(opts.Gamma))
	epsilon := gorgonia.NewConstant(float32(opts.Epsilon))

	tabNetLoss := gorgonia.NewScalar(nn.TrainGraph(), tensor.Float32, gorgonia.WithValue(float32(0.0)), gorgonia.WithName("TabNetLoss"))
	stepsCount := gorgonia.NewScalar(nn.TrainGraph(), tensor.Float32, gorgonia.WithValue(float32(opts.DecisionSteps)), gorgonia.WithName("Steps"))

	prior := gorgonia.NewTensor(nn.TrainGraph(), tensor.Float32, 2, gorgonia.WithShape(opts.BatchSize, opts.InputSize), gorgonia.WithInit(gorgonia.Ones()), gorgonia.WithName("Prior"))
	out := gorgonia.NewTensor(nn.TrainGraph(), tensor.Float32, 2, gorgonia.WithShape(opts.BatchSize, opts.PredictionLayerDim), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName("Output"))

	return func(nodes ...*gorgonia.Node) (godl.Result, error) {
		x := nodes[0]

		bn, err := bnLayer(x)
		if err != nil {
			return godl.Result{}, fmt.Errorf("applying initial batch norm %v: %w", x.Shape(), err)
		}

		ft, err := initialSplitter(bn.Output)
		if err != nil {
			return godl.Result{}, fmt.Errorf("applying initial splitter %v: %w", bn.Output.Shape(), err)
		}

		xAttentiveLayer, err := gorgonia.Slice(ft.Output, nil, gorgonia.S(opts.PredictionLayerDim, ft.Shape()[1]))
		if err != nil {
			return godl.Result{}, fmt.Errorf("slicing %v: %w", ft.Shape(), err)
		}

		for i := 0; i < opts.DecisionSteps; i++ {
			attentiveTransformer := attentiveTransformers[i]
			featureTransformer := featureTransformers[i]

			// nn.Watch("prior", prior)

			result, err := attentiveTransformer(xAttentiveLayer, prior)
			if err != nil {
				return godl.Result{}, err
			}

			mask := result.Output

			stepLoss := gorgonia.Must(gorgonia.Mean(
				gorgonia.Must(gorgonia.Sum(
					gorgonia.Must(gorgonia.HadamardProd(
						mask,
						gorgonia.Must(gorgonia.Log(
							gorgonia.Must(gorgonia.Add(mask, epsilon)),
						)),
					)),
					1,
				)),
			))

			// accum losses
			tabNetLoss = gorgonia.Must(gorgonia.Add(tabNetLoss, stepLoss))

			// Update prior
			{
				prior, err = gorgonia.HadamardProd(gorgonia.Must(gorgonia.Sub(gamma, mask)), prior)
				if err != nil {
					return godl.Result{}, fmt.Errorf("updating prior: %w", err)
				}
			}

			maskedX, err := gorgonia.HadamardProd(mask, bn.Output)
			if err != nil {
				return godl.Result{}, err
			}

			ds, err := featureTransformer(maskedX)
			if err != nil {
				return godl.Result{}, err
			}

			firstPart, err := gorgonia.Slice(ds.Output, nil, gorgonia.S(0, opts.PredictionLayerDim))
			if err != nil {
				return godl.Result{}, err
			}

			relu, err := gorgonia.Rectify(firstPart)
			if err != nil {
				return godl.Result{}, err
			}

			out, err = gorgonia.Add(out, relu)
			if err != nil {
				return godl.Result{}, err
			}

			xAttentiveLayer, err = gorgonia.Slice(ds.Output, nil, gorgonia.S(opts.PredictionLayerDim, ds.Shape()[1]))
			if err != nil {
				return godl.Result{}, err
			}
		}

		tabNetLoss = gorgonia.Must(gorgonia.Div(tabNetLoss, stepsCount))

		// nn.Watch("out before", out)

		result, err := finalMapping(out)
		if err != nil {
			return godl.Result{}, fmt.Errorf("TabNet: applying final FC layer to %v: %w", out.Shape(), err)
		}

		// nn.Watch("out after", result.Output)

		result.Loss = tabNetLoss

		return result, nil
	}
}
