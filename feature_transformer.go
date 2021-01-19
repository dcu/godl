package tabnet

import (
	"fmt"
	"math"

	"gorgonia.org/gorgonia"
)

// FeatureTransformerOpts contains options for feature transformer layer
type FeatureTransformerOpts struct {
	Shared            []Layer
	VirtualBatchSize  int
	IndependentBlocks int
	InputDimension    int
	OutputDimension   int
	Inferring         bool
	WithBias          bool

	WeightsInit gorgonia.InitWFn
}

// FeatureTransformer implements a feature transformer layer
func (nn *Model) FeatureTransformer(opts FeatureTransformerOpts) Layer {
	if opts.InputDimension == 0 {
		panic("input dimension can't be nil")
	}

	if opts.OutputDimension == 0 {
		panic("output dimension can't be nil")
	}

	shared := make([]Layer, 0, len(opts.Shared))

	gluInput := opts.InputDimension
	gluOutput := opts.OutputDimension

	for _, fcLayer := range opts.Shared {
		shared = append(shared, nn.GLU(GLUOpts{
			InputDimension:   gluInput,
			OutputDimension:  gluOutput,
			VirtualBatchSize: opts.VirtualBatchSize,
			FC:               fcLayer,
			WeightsInit:      opts.WeightsInit,
			Inferring:        opts.Inferring,
			WithBias:         opts.WithBias,
		}))

		gluInput = gluOutput
	}

	independentBlocks := opts.IndependentBlocks
	gluOutput = opts.OutputDimension

	independent := make([]Layer, 0, len(opts.Shared))
	for i := 0; i < independentBlocks; i++ {
		independent = append(independent, nn.GLU(GLUOpts{
			OutputDimension:  gluOutput,
			InputDimension:   gluInput,
			VirtualBatchSize: opts.VirtualBatchSize,
			WeightsInit:      opts.WeightsInit,
			WithBias:         opts.WithBias,
		}))
	}

	scale := gorgonia.NewConstant(math.Sqrt(0.5))

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, error) {
		var err error
		x := nodes[0]

		if len(shared) > 0 {
			x, _, err = shared[0](x)
			if err != nil {
				return nil, nil, err
			}

			for _, glu := range shared[1:] {
				output, _, err := glu(x)
				if err != nil {
					return nil, nil, fmt.Errorf("AttentiveTransformer: executing shared GLU layer with %v: %w", x.Shape(), err)
				}

				xShape := x.Shape()
				x, err = gorgonia.Add(x, output)
				if err != nil {
					return nil, nil, fmt.Errorf("AttentiveTransformer: %v + %v: %w", xShape, output.Shape(), err)
				}

				x, err = gorgonia.Mul(x, scale)
				if err != nil {
					return nil, nil, err
				}
			}
		}

		for _, layer := range independent {
			output, _, err := layer(x)
			if err != nil {
				return nil, nil, fmt.Errorf("AttentiveTransformer: executing independent GLU layer with %v: %w", x.Shape(), err)
			}

			x, err = gorgonia.Add(x, output)
			if err != nil {
				return nil, nil, err
			}

			x, err = gorgonia.Mul(x, scale)
			if err != nil {
				return nil, nil, err
			}
		}

		return x, nil, nil
	}
}
