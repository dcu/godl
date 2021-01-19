package tabnet

import (
	"fmt"
	"sync/atomic"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var fcCount uint64 = 0

// FCOpts contains optional parameter for a layer
type FCOpts struct {
	Activation      ActivationFn
	Dropout         float64
	OutputDimension int
	InputDimension  int

	WeightsInit gorgonia.InitWFn
	WithBias    bool
}

func (nn *Model) FC(opts FCOpts) Layer {
	if opts.InputDimension == 0 {
		panic("input dimension must be set")
	}

	if opts.OutputDimension == 0 {
		panic("output dimension must be set")
	}

	atomic.AddUint64(&fcCount, 1)

	layerType := fmt.Sprintf("FC_%d", fcCount)

	var (
		bias *gorgonia.Node
		w    = nn.addWeights(layerType, tensor.Shape{opts.InputDimension, opts.OutputDimension}, opts.WeightsInit)
	)

	if opts.WithBias {
		bias = nn.addBias(layerType, tensor.Shape{1, opts.OutputDimension}, opts.WeightsInit)
	}

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, error) {
		x := nodes[0]
		xShape := x.Shape()

		if opts.OutputDimension <= 0 {
			return nil, nil, fmt.Errorf("%s: wrong output features count %d for FC layer", layerType, opts.OutputDimension)
		}

		if x.Dims() > 2 {
			b, v := xShape[0], tensor.Shape(xShape[1:]).TotalSize()
			x = gorgonia.Must(gorgonia.Reshape(x, tensor.Shape{b, v}))
		}

		layer, err := gorgonia.Mul(x, w)
		if err != nil {
			return nil, nil, fmt.Errorf("%s: error applying mul %v x %v: %w ", layerType, x.Shape(), w.Shape(), err)
		}

		if opts.WithBias {
			layer, err = gorgonia.BroadcastAdd(layer, bias, nil, []byte{0})
			if err != nil {
				return nil, nil, fmt.Errorf("%s: error adding bias %w", layerType, err)
			}
		}

		if opts.Activation != nil {
			layer, err = opts.Activation(layer)
			if err != nil {
				return nil, nil, fmt.Errorf("%s: error applying activation %w", layerType, err)
			}
		}

		if opts.Dropout > 0.0 {
			layer, err = gorgonia.Dropout(layer, opts.Dropout)
			if err != nil {
				return nil, nil, fmt.Errorf("%s: error applying dropout %w", layerType, err)
			}
		}

		return layer, nil, nil
	}
}
