package tabnet

import "gorgonia.org/gorgonia"

type DecisionStepOpts struct {
	Shared            []Layer
	VirtualBatchSize  int
	IndependentBlocks int
}

func (nn *TabNet) DecisionStep(prior *gorgonia.Node, opts FeatureTransformerOpts) Layer {
	return func(x *gorgonia.Node) (*gorgonia.Node, error) {
		mask, err := nn.AttentiveTransformer(x, prior, FCOpts{}, GBNOpts{})
		if err != nil {
			return nil, err
		}

		mul, err := gorgonia.HadamardProd(x, mask)
		if err != nil {
			return nil, err
		}

		ft, err := nn.FeatureTransformer(opts)(mul)
		if err != nil {
			return nil, err
		}

		return ft, nil
	}
}
