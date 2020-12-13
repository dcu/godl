package tabnet

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// FCOpts contains optional parameter for a layer
type FCOpts struct {
	ActivationFn   ActivationFn
	Dropout        float64
	OutputFeatures int
	WeightsInit    gorgonia.InitWFn
	// TODO: support bias
}

func (nn *Model) FC(opts FCOpts) Layer {
	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, error) {
		x := nodes[0]
		xShape := x.Shape()

		if opts.OutputFeatures <= 0 {
			return nil, fmt.Errorf("wrong output features count %d for FC layer", opts.OutputFeatures)
		}

		if x.Dims() > 2 {
			b, v := xShape[0], tensor.Shape(xShape[1:]...).TotalSize()
			x = gorgonia.Must(gorgonia.Reshape(x, tensor.Shape{b, v}))
		}

		shape := tensor.Shape{x.Shape()[1], opts.OutputFeatures}
		layerNumber := nn.LayersCount() + 1

		log.Printf("Layer %d: FC(%v,%v)", layerNumber, x.Shape(), shape)

		w := nn.addWeights(shape, opts.WeightsInit)
		layer, err := gorgonia.Mul(x, w)
		if err != nil {
			return nil, fmt.Errorf("Layer %d: error applying mul %w", layerNumber, err)
		}

		if opts.ActivationFn != nil {
			layer, err = opts.ActivationFn(layer)
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
