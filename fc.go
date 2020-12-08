package tabnet

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Layer func(x *gorgonia.Node) (*gorgonia.Node, error)

func (nn *TabNet) FC(opts FCOpts) Layer {
	return func(x *gorgonia.Node) (*gorgonia.Node, error) {
		if opts.OutputFeatures <= 0 {
			return nil, fmt.Errorf("wrong output features count %d for FC layer", opts.OutputFeatures)
		}

		if x.Dims() == 4 {
			b, c, h, w := x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
			x = gorgonia.Must(gorgonia.Reshape(x, tensor.Shape{b, c * h * w}))
		}

		shape := tensor.Shape{x.Shape()[1], opts.OutputFeatures}

		log.Printf("Layer %d: FC(%v,%v)", nn.LayersCount()+1, x.Shape(), shape)

		w := nn.addWeights(shape)
		layer, err := gorgonia.Mul(x, w)
		if err != nil {
			return nil, err
		}

		if opts.ActivationFn != nil {
			layer, err = opts.ActivationFn(layer)
			if err != nil {
				return nil, err
			}
		}

		if opts.Dropout > 0.0 {
			layer, err = gorgonia.Dropout(layer, opts.Dropout)
			if err != nil {
				return nil, err
			}
		}

		return layer, nil
	}
}
