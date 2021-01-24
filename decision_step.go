package tabnet

import (
	"gorgonia.org/gorgonia"
)

type DecisionStepOpts struct {
	Shared []Layer

	IndependentBlocks int

	PredictionLayerDim int
	AttentionLayerDim  int

	InputDimension  int
	OutputDimension int

	MaskFunction ActivationFn

	WithBias                         bool
	Momentum                         float64
	Epsilon                          float64
	VirtualBatchSize                 int
	Inferring                        bool
	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

type DecisionStep struct {
	Name                 layerType
	FeatureTransformer   Layer
	AttentiveTransformer Layer
}

func (step DecisionStep) CalculateMask(xAttentiveLayer, prior, epsilon *gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, error) {
	mask, _, err := step.AttentiveTransformer(xAttentiveLayer, prior)
	if err != nil {
		return nil, nil, errorF(step.Name, "attentive transformer: %w", err)
	}

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

func (nn *Model) DecisionStep(opts DecisionStepOpts) *DecisionStep {
	lt := incLayer("DecisionStep")

	mustBeGreaterThan(lt, "InputDimension", opts.InputDimension, 0)
	mustBeGreaterThan(lt, "OutputDimension", opts.OutputDimension, 0)

	ds := &DecisionStep{
		Name: lt,
	}

	ds.FeatureTransformer = nn.FeatureTransformer(FeatureTransformerOpts{
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

	ds.AttentiveTransformer = nn.AttentiveTransformer(AttentiveTransformerOpts{
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
