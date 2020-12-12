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
	OutputFeatures    int
}

// FeatureTransformer implements a feature transformer layer
func (nn *Model) FeatureTransformer(opts FeatureTransformerOpts) Layer {
	shared := make([]Layer, 0, len(opts.Shared))
	for _, fcLayer := range opts.Shared {
		shared = append(shared, nn.GLU(GLUOpts{
			VirtualBatchSize: opts.VirtualBatchSize,
			FC:               fcLayer,
			OutputFeatures:   opts.OutputFeatures,
		}))
	}

	independent := make([]Layer, 0, len(opts.Shared))
	for i := 0; i < opts.IndependentBlocks; i++ {
		independent = append(independent, nn.GLU(GLUOpts{
			OutputFeatures:   opts.OutputFeatures,
			VirtualBatchSize: opts.VirtualBatchSize,
		}))
	}

	scale := gorgonia.NewScalar(nn.g, tensor.Float64, gorgonia.WithValue(math.Sqrt(0.5))) // TODO: make configurable

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, error) {
		x := nodes[0]

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
