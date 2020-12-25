package tabnet

import (
	"fmt"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// FCOpts contains optional parameter for a layer
type FCOpts struct {
	Activation      ActivationFn
	Dropout         float64
	OutputDimension int
	WeightsInit     gorgonia.InitWFn
	WithBias        bool
}

func (nn *Model) FC(opts FCOpts) Layer {
	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, error) {
		x := nodes[0]
		xShape := x.Shape()

		if opts.OutputDimension <= 0 {
			return nil, fmt.Errorf("wrong output features count %d for FC layer", opts.OutputDimension)
		}

		if x.Dims() > 2 {
			b, v := xShape[0], tensor.Shape(xShape[1:]).TotalSize()
			x = gorgonia.Must(gorgonia.Reshape(x, tensor.Shape{b, v}))
		}

		shape := tensor.Shape{x.Shape()[1], opts.OutputDimension}
		layerNumber := nn.LearnablesCount() + 1

		w := nn.addWeights(shape, opts.WeightsInit)
		layer, err := gorgonia.Mul(x, w)
		if err != nil {
			return nil, fmt.Errorf("Layer %d: error applying mul %w", layerNumber, err)
		}

		if opts.WithBias {
			b := nn.addBias(shape, opts.WeightsInit)

			layer, err = gorgonia.Add(layer, b)
			if err != nil {
				return nil, fmt.Errorf("Layer %d: error adding bias %w", layerNumber, err)
			}
		}

		if opts.Activation != nil {
			layer, err = opts.Activation(layer)
			if err != nil {
				return nil, fmt.Errorf("Layer %d: error applying activation %w", layerNumber, err)
			}
		}

		if opts.Dropout > 0.0 {
			layer, err = gorgonia.Dropout(layer, opts.Dropout)
			if err != nil {
				return nil, fmt.Errorf("Layer %d: error applying dropout %w", layerNumber, err)
			}
		}

		return layer, nil
	}
}
