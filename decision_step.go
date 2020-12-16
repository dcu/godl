package tabnet

import (
	"gorgonia.org/gorgonia"
)

type DecisionStepOpts struct {
	Shared []Layer

	IndependentBlocks int

	PredictionLayerDim int
	AttentionLayerDim  int

	Momentum                         float64
	Epsilon                          float64
	VirtualBatchSize                 int
	Inferring                        bool
	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

func (nn *Model) DecisionStep(opts DecisionStepOpts) Layer {
	featureTransformer := nn.FeatureTransformer(FeatureTransformerOpts{
		Shared:            opts.Shared,
		VirtualBatchSize:  opts.VirtualBatchSize,
		OutputFeatures:    opts.AttentionLayerDim + opts.PredictionLayerDim,
		IndependentBlocks: opts.IndependentBlocks,
		WeightsInit:       opts.WeightsInit,
	})

	attentiveTransformer := nn.AttentiveTransformer(AttentiveTransformerOpts{
		OutputFeatures:   opts.AttentionLayerDim + opts.PredictionLayerDim,
		Momentum:         opts.Momentum,
		Epsilon:          opts.Epsilon,
		VirtualBatchSize: opts.VirtualBatchSize,
		Inferring:        opts.Inferring,
		ScaleInit:        opts.ScaleInit,
		BiasInit:         opts.BiasInit,
		WeightsInit:      opts.WeightsInit,
	})

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, error) {
		if err := nn.checkArity("DecisionStep", nodes, 3); err != nil {
			return nil, err
		}

		x := nodes[0]
		xAttentiveLayer := nodes[1]
		prior := nodes[2]

		mask, err := attentiveTransformer(xAttentiveLayer, prior)
		if err != nil {
			return nil, err
		}

		mul, err := hadamardProd(x, mask)
		if err != nil {
			return nil, err
		}

		ft, err := featureTransformer(mul)
		if err != nil {
			return nil, err
		}

		return ft, nil
	}
}
