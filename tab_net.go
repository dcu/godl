package tabnet

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// TabNetOpts contains parameters to configure the tab net algorithm
type TabNetOpts struct {
	OutputFeatures    int
	SharedBlocks      int
	IndependentBlocks int
	DecisionSteps     int

	PredictionLayerDim int
	AttentionLayerDim  int

	Momentum            float64
	Epsilon             float64
	VirtualBatchSize    int
	Inferring           bool
	ScaleInit, BiasInit gorgonia.InitWFn
}

// TabNet implements the tab net architecture
func (nn *Model) TabNet(opts TabNetOpts) Layer {
	shared := make([]Layer, 0, opts.SharedBlocks)

	for i := 0; i < opts.SharedBlocks; i++ {
		shared = append(shared, nn.FC(FCOpts{
			OutputFeatures: 2 * (opts.PredictionLayerDim + opts.AttentionLayerDim),
		}))
	}

	steps := make([]Layer, 0, opts.DecisionSteps)
	for i := 0; i < opts.DecisionSteps; i++ {
		steps = append(steps, nn.DecisionStep(
			DecisionStepOpts{
				Shared:            shared,
				IndependentBlocks: opts.IndependentBlocks,
				VirtualBatchSize:  opts.VirtualBatchSize,
			},
		))
	}

	fcLayer := nn.FC(FCOpts{
		OutputFeatures: opts.OutputFeatures,
	})

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, error) {
		x := nodes[0]
		xShape := x.Shape()

		bn, err := nn.BN(BNOpts{})(x) // TODO: make configurable
		if err != nil {
			return nil, err
		}

		ft, err := nn.FeatureTransformer(FeatureTransformerOpts{
			Shared:            shared,
			VirtualBatchSize:  opts.VirtualBatchSize,
			IndependentBlocks: opts.IndependentBlocks,
			OutputFeatures:    opts.AttentionLayerDim + opts.PredictionLayerDim,
		})(bn)
		if err != nil {
			return nil, err
		}

		xAttentiveLayer, err := gorgonia.Slice(ft, nil, gorgonia.S(opts.PredictionLayerDim, ft.Shape()[1]))
		if err != nil {
			return nil, err
		}

		prior := gorgonia.NewTensor(nn.g, tensor.Float64, bn.Shape().Dims(), gorgonia.WithShape(bn.Shape()...), gorgonia.WithInit(gorgonia.Ones()))
		out := gorgonia.NewTensor(nn.g, tensor.Float64, 3, gorgonia.WithShape(xShape[0], xShape[1], opts.PredictionLayerDim), gorgonia.WithInit(gorgonia.Zeroes()))

		for _, step := range steps {
			ds, err := step(bn, xAttentiveLayer, prior)
			if err != nil {
				return nil, err
			}

			ds, err = gorgonia.Slice(ds, nil, nil, gorgonia.S(0, opts.IndependentBlocks))
			if err != nil {
				return nil, err
			}

			relu, err := gorgonia.Rectify(ds)
			if err != nil {
				return nil, err
			}

			out, err = gorgonia.Add(out, relu)
			if err != nil {
				return nil, err
			}

			xAttentiveLayer, err = gorgonia.Slice(ds, nil, nil, gorgonia.S(opts.IndependentBlocks, ds.Shape()[2]))
			if err != nil {
				return nil, err
			}
		}

		return fcLayer(out)
	}
}
