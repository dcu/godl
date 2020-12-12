package tabnet

import "gorgonia.org/gorgonia"

type DecisionStepOpts struct {
	Shared []Layer

	IndependentBlocks int

	PredictionLayerDim int
	AttentionLayerDim  int

	Momentum            float64
	Epsilon             float64
	VirtualBatchSize    int
	Inferring           bool
	ScaleInit, BiasInit gorgonia.InitWFn
}

func (nn *Model) DecisionStep(opts DecisionStepOpts) Layer {
	featureTransformer := nn.FeatureTransformer(FeatureTransformerOpts{
		Shared:            opts.Shared,
		VirtualBatchSize:  opts.VirtualBatchSize,
		OutputFeatures:    opts.AttentionLayerDim + opts.PredictionLayerDim,
		IndependentBlocks: opts.IndependentBlocks,
	})

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, error) {
		x := nodes[0]
		xAttentiveLayer := nodes[1]
		prior := nodes[2]

		attentiveTransformer := nn.AttentiveTransformer(AttentiveTransformerOpts{
			OutputFeatures:   x.Shape()[1],
			Momentum:         opts.Momentum,
			Epsilon:          opts.Epsilon,
			VirtualBatchSize: opts.VirtualBatchSize,
			Inferring:        opts.Inferring,
			ScaleInit:        opts.ScaleInit,
			BiasInit:         opts.BiasInit,
		})

		mask, err := attentiveTransformer(xAttentiveLayer, prior)
		if err != nil {
			return nil, err
		}

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
}
