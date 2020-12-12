package tabnet

import "gorgonia.org/gorgonia"

type DecisionStepOpts struct {
	Shared            []Layer
	VirtualBatchSize  int
	IndependentBlocks int
}

func (nn *Model) DecisionStep(opts DecisionStepOpts) Layer {
	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, error) {
		x := nodes[0]
		size := nodes[1]
		prior := nodes[2]

		_ = size

		mask, err := nn.AttentiveTransformer(x, prior, FCOpts{}, GBNOpts{})
		if err != nil {
			return nil, err
		}

		mul, err := gorgonia.HadamardProd(x, mask)
		if err != nil {
			return nil, err
		}

		ft, err := nn.FeatureTransformer(FeatureTransformerOpts{
			Shared: opts.Shared,
			VirtualBatchSize: opts.VirtualBatchSize,
			IndependentBlocks: ,
		})(mul)
		if err != nil {
			return nil, err
		}

		return ft, nil
	}
}
