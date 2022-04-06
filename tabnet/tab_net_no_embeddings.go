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

type TabNetNoEmbeddingsModule struct {
	model *godl.Model
	opts  TabNetNoEmbeddingsOpts

	bn              *godl.BatchNormModule
	initialSplitter *FeatureTransformerModule
	finalMapping    *godl.LinearModule

	attentiveTransformers []*AttentiveTransformerModule
	featureTransformers   []*FeatureTransformerModule
}

func (m *TabNetNoEmbeddingsModule) Forward(inputs ...*godl.Node) godl.Nodes {
	x := inputs[0]

	bn := m.bn.Forward(x)[0]
	ft := m.initialSplitter.Forward(bn)[0]

	xAttentiveLayer := gorgonia.Must(gorgonia.Slice(ft, nil, gorgonia.S(m.opts.PredictionLayerDim, ft.Shape()[1])))

	gamma := gorgonia.NewConstant(float32(m.opts.Gamma))
	epsilon := gorgonia.NewConstant(float32(m.opts.Epsilon))

	loss := gorgonia.NewScalar(m.model.TrainGraph(), tensor.Float32, gorgonia.WithValue(float32(0.0)), gorgonia.WithName("TabNetLoss"))
	stepsCount := gorgonia.NewScalar(m.model.TrainGraph(), tensor.Float32, gorgonia.WithValue(float32(m.opts.DecisionSteps)), gorgonia.WithName("Steps"))

	prior := gorgonia.NewTensor(m.model.TrainGraph(), tensor.Float32, 2, gorgonia.WithShape(m.opts.BatchSize, m.opts.InputSize), gorgonia.WithInit(gorgonia.Ones()), gorgonia.WithName("Prior"))
	out := gorgonia.NewTensor(m.model.TrainGraph(), tensor.Float32, 2, gorgonia.WithShape(m.opts.BatchSize, m.opts.PredictionLayerDim), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName("Output"))

	for i := 0; i < m.opts.DecisionSteps; i++ {
		attentiveTransformer := m.attentiveTransformers[i]
		featureTransformer := m.featureTransformers[i]

		result := attentiveTransformer.Forward(xAttentiveLayer, prior)

		mask := result[0]

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
		loss = gorgonia.Must(gorgonia.Add(loss, stepLoss))

		// Update prior
		{
			prior = gorgonia.Must(gorgonia.HadamardProd(gorgonia.Must(gorgonia.Sub(gamma, mask)), prior))
		}

		maskedX := gorgonia.Must(gorgonia.HadamardProd(mask, bn))

		ds := featureTransformer.Forward(maskedX)[0]

		firstPart := gorgonia.Must(gorgonia.Slice(ds, nil, gorgonia.S(0, m.opts.PredictionLayerDim)))

		relu := gorgonia.Must(gorgonia.Rectify(firstPart))

		out = gorgonia.Must(gorgonia.Add(out, relu))

		xAttentiveLayer = gorgonia.Must(gorgonia.Slice(ds, nil, gorgonia.S(m.opts.PredictionLayerDim, ds.Shape()[1])))
	}

	loss = gorgonia.Must(gorgonia.Div(loss, stepsCount))
	result := m.finalMapping.Forward(out)[0]

	return godl.Nodes{result, loss}
}

// TabNetNoEmbeddings implements the tab net architecture
func TabNetNoEmbeddings(nn *godl.Model, opts TabNetNoEmbeddingsOpts) *TabNetNoEmbeddingsModule {
	opts.setDefaults()

	bnLayer := godl.BatchNorm1d(nn, godl.BatchNormOpts{
		ScaleInit: opts.ScaleInit,
		BiasInit:  opts.BiasInit,
		InputSize: opts.InputSize,
		Momentum:  0.01,
	})

	shared := make([]*godl.LinearModule, 0, opts.SharedBlocks)
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

			shared = append(shared, godl.Linear(nn, godl.LinearOpts{
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

	featureTransformers := make([]*FeatureTransformerModule, 0, opts.DecisionSteps)
	attentiveTransformers := make([]*AttentiveTransformerModule, 0, opts.DecisionSteps)

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

	finalMapping := godl.Linear(nn, godl.LinearOpts{
		InputDimension:  opts.PredictionLayerDim,
		OutputDimension: opts.OutputSize,
		WeightsInit:     weightsInit,
		WeightsName:     "FinalMapping",
		WithBias:        opts.WithBias,
	})

	return &TabNetNoEmbeddingsModule{
		model:                 nn,
		opts:                  opts,
		bn:                    bnLayer,
		initialSplitter:       initialSplitter,
		finalMapping:          finalMapping,
		attentiveTransformers: attentiveTransformers,
		featureTransformers:   featureTransformers,
	}
}
