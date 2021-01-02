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

	Momentum                         float64
	Epsilon                          float64
	VirtualBatchSize                 int
	Inferring                        bool
	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

type DecisionStep struct {
	FeatureTransformer   Layer
	AttentiveTransformer Layer
}

func (nn *Model) DecisionStep(opts DecisionStepOpts) *DecisionStep {
	if opts.OutputDimension == 0 {
		panic("OutputDimension must be set")
	}

	ds := &DecisionStep{}

	ds.AttentiveTransformer = nn.AttentiveTransformer(AttentiveTransformerOpts{
		InputDimension:   opts.InputDimension,
		OutputDimension:  opts.OutputDimension,
		Momentum:         opts.Momentum,
		Epsilon:          opts.Epsilon,
		VirtualBatchSize: opts.VirtualBatchSize,
		Inferring:        opts.Inferring,
		ScaleInit:        opts.ScaleInit,
		BiasInit:         opts.BiasInit,
		WeightsInit:      opts.WeightsInit,
		Activation:       opts.MaskFunction,
	})

	featureTransformer := nn.FeatureTransformer(FeatureTransformerOpts{
		Shared:            opts.Shared,
		VirtualBatchSize:  opts.VirtualBatchSize,
		OutputDimension:   opts.AttentionLayerDim + opts.PredictionLayerDim,
		IndependentBlocks: opts.IndependentBlocks,
		WeightsInit:       opts.WeightsInit,
		Inferring:         opts.Inferring,
	})

	ds.FeatureTransformer = func(nodes ...*gorgonia.Node) (*gorgonia.Node, error) {
		if err := nn.checkArity("DecisionStep-FeatureTransformer", nodes, 2); err != nil {
			return nil, err
		}

		x := nodes[0]
		mask := nodes[1]

		mul, err := gorgonia.HadamardProd(x, mask)
		if err != nil {
			return nil, err
		}

		ft, err := featureTransformer(mul)
		if err != nil {
			return nil, err
		}

		return ft, nil
	}

	return ds
}
