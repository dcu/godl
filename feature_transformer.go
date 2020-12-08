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
}

// FeatureTransformer implements a feature transformer layer
func (nn *TabNet) FeatureTransformer(x *gorgonia.Node, opts FeatureTransformerOpts) Layer {
	shared := make([]Layer, 0, len(opts.Shared))
	for _, fcLayer := range opts.Shared {
		shared = append(shared, nn.GLU(GLUOpts{
			VirtualBatchSize: opts.VirtualBatchSize,
			FC:               fcLayer,
		}))
	}

	independent := make([]Layer, 0, len(opts.Shared))
	for i := 0; i < opts.IndependentBlocks; i++ {
		independent = append(independent, nn.GLU(GLUOpts{
			VirtualBatchSize: opts.VirtualBatchSize,
		}))
	}

	scale := gorgonia.NewScalar(nn.g, tensor.Float64, gorgonia.WithValue(math.Sqrt(0.5))) // TODO: make configurable

	return func(x *gorgonia.Node) (*gorgonia.Node, error) {
		for _, layer := range shared {
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
