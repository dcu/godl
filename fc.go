package tabnet

import (
	"log"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func (nn *TabNet) FC(x *gorgonia.Node, shape tensor.Shape, opts FCOpts) *gorgonia.Node {
	if x.Dims() == 4 {
		b, c, h, w := x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
		x = gorgonia.Must(gorgonia.Reshape(x, tensor.Shape{b, c * h * w}))
	}

	log.Printf("Layer %d: FC(%v,%v)", nn.LayersCount()+1, x.Shape(), shape)

	w := nn.addWeights(shape)
	layer := gorgonia.Must(gorgonia.Mul(x, w))

	if opts.ActivationFn != nil {
		layer = gorgonia.Must(opts.ActivationFn(layer))
	}

	if opts.Dropout > 0.0 {
		layer = gorgonia.Must(gorgonia.Dropout(layer, opts.Dropout))
	}

	return layer
}
