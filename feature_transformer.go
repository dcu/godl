package tabnet

import (
	"math"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// FeatureTransformerOpts contains options for feature transformer layer
type FeatureTransformerOpts struct {
	Shared            []Layer
	VirtualBatchSize  int
	IndependentBlocks int
	OutputDimension    int
	Inferring         bool

	WeightsInit gorgonia.InitWFn
}

// FeatureTransformer implements a feature transformer layer
func (nn *Model) FeatureTransformer(opts FeatureTransformerOpts) Layer {
	shared := make([]Layer, 0, len(opts.Shared))
	for _, fcLayer := range opts.Shared {
		shared = append(shared, nn.GLU(GLUOpts{
			VirtualBatchSize: opts.VirtualBatchSize,
			FC:               fcLayer,
			OutputDimension:   opts.OutputDimension,
			WeightsInit:      opts.WeightsInit,
			Inferring:        opts.Inferring,
		}))
	}

	independent := make([]Layer, 0, len(opts.Shared))
	for i := 0; i < opts.IndependentBlocks; i++ {

		independent = append(independent, nn.GLU(GLUOpts{
			OutputDimension:   opts.OutputDimension,
			VirtualBatchSize: opts.VirtualBatchSize,
			WeightsInit:      opts.WeightsInit,
		}))
	}

	scale := gorgonia.NewScalar(nn.g, tensor.Float64, gorgonia.WithValue(math.Sqrt(0.5))) // TODO: make configurable

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, error) {
		var err error
		x := nodes[0]

		if len(shared) > 0 {
			x, err = shared[0](x)
			if err != nil {
				return nil, err
			}

			for _, layer := range shared[1:] {
				output, err := layer(x)
				if err != nil {
					return nil, err
				}

				x, err = gorgonia.Add(x, output)
				if err != nil {
					return nil, err
				}

				x, err = gorgonia.Mul(x, scale)
				if err != nil {
					return nil, err
				}
			}
		}

		for _, layer := range independent {
			output, err := layer(x)
			if err != nil {
				return nil, err
			}

			x, err = gorgonia.Add(x, output)
			if err != nil {
				return nil, err
			}

			x, err = gorgonia.Mul(x, scale)
			if err != nil {
				return nil, err
			}
		}

		return x, nil
	}
}
