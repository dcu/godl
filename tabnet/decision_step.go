package tabnet

import (
	"github.com/dcu/deepzen"
	"gorgonia.org/gorgonia"
)

type DecisionStepOpts struct {
	Shared []deepzen.Layer

	IndependentBlocks int

	PredictionLayerDim int
	AttentionLayerDim  int

	InputDimension  int
	OutputDimension int

	MaskFunction deepzen.ActivationFn

	WithBias                         bool
	Momentum                         float32
	Epsilon                          float32
	VirtualBatchSize                 int
	Inferring                        bool
	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

type DecisionStep struct {
	Name                 deepzen.LayerType
	FeatureTransformer   deepzen.Layer
	AttentiveTransformer deepzen.Layer
}

func (step DecisionStep) CalculateMask(xAttentiveLayer, prior, epsilon *gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, error) {
	result, err := step.AttentiveTransformer(xAttentiveLayer, prior)
	if err != nil {
		return nil, nil, deepzen.ErrorF(step.Name, "attentive transformer: %w", err)
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

	return mask, stepLoss, nil
}

func NewDecisionStep(nn *deepzen.Model, opts DecisionStepOpts) *DecisionStep {
	lt := deepzen.AddLayer("DecisionStep")

	deepzen.MustBeGreatherThan(lt, "InputDimension", opts.InputDimension, 0)
	deepzen.MustBeGreatherThan(lt, "OutputDimension", opts.OutputDimension, 0)

	ds := &DecisionStep{
		Name: lt,
	}

	ds.FeatureTransformer = FeatureTransformer(nn, FeatureTransformerOpts{
		Shared:            opts.Shared,
		VirtualBatchSize:  opts.VirtualBatchSize,
		InputDimension:    opts.OutputDimension,
		OutputDimension:   opts.AttentionLayerDim + opts.PredictionLayerDim,
		IndependentBlocks: opts.IndependentBlocks,
		WeightsInit:       opts.WeightsInit,
		Inferring:         opts.Inferring,
		WithBias:          opts.WithBias,
		Momentum:          opts.Momentum,
	})

	ds.AttentiveTransformer = AttentiveTransformer(nn, AttentiveTransformerOpts{
		InputDimension:   opts.PredictionLayerDim,
		OutputDimension:  opts.OutputDimension,
		Momentum:         opts.Momentum,
		Epsilon:          opts.Epsilon,
		VirtualBatchSize: opts.VirtualBatchSize,
		Inferring:        opts.Inferring,
		ScaleInit:        opts.ScaleInit,
		BiasInit:         opts.BiasInit,
		WeightsInit:      opts.WeightsInit,
		Activation:       opts.MaskFunction,
		WithBias:         opts.WithBias,
	})

	return ds
}
