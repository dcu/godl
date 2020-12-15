package tabnet

import (
	"fmt"
	"log"
	"math"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// GBNOpts contains config options for the ghost batch normalization
type GBNOpts struct {
	Momentum                         float64
	Epsilon                          float64
	VirtualBatchSize                 int
	Inferring                        bool
	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

func (o *GBNOpts) setDefaults() {
	if o.VirtualBatchSize == 0 {
		o.VirtualBatchSize = 128
	}

	if o.Momentum == 0.0 {
		o.Momentum = 0.01
	}

	if o.Epsilon == 0.0 {
		o.Epsilon = 1e-5
	}

	if o.ScaleInit == nil {
		o.ScaleInit = gorgonia.Ones()
	}

	if o.BiasInit == nil {
		o.BiasInit = gorgonia.Zeroes()
	}
}

// GBN implements a Ghost Batch Normalization: https://arxiv.org/pdf/1705.08741.pdf
// momentum defaults to 0.01 if 0 is passed
// epsilon defaults to 1e-5 if 0 is passed
func (nn *Model) GBN(opts GBNOpts) Layer {
	opts.setDefaults()

	return func(inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
		if err := nn.checkArity("GBN", inputs, 1); err != nil {
			return nil, err
		}

		x := inputs[0]
		xShape := x.Shape()
		inputSize := xShape.TotalSize()

		if x.Dims() > 1 {
			x = gorgonia.Must(gorgonia.Reshape(x, tensor.Shape{inputSize}))
		}

		if opts.VirtualBatchSize > inputSize {
			opts.VirtualBatchSize = inputSize
		}

		if inputSize%opts.VirtualBatchSize != 0 {
			panic(fmt.Errorf("input size (%d) must be divisable by virtual batch size (%v)", inputSize, opts.VirtualBatchSize))
		}

		batches := int(math.Ceil(float64(inputSize) / float64(opts.VirtualBatchSize)))
		nodes := make([]*gorgonia.Node, 0, batches)

		log.Printf("GBN %v", x.Shape())

		// Split the vector in virtual batches
		for vb := 0; vb < batches; vb++ {
			start := vb * opts.VirtualBatchSize
			if start > inputSize {
				break
			}

			end := start + opts.VirtualBatchSize
			if end > inputSize {
				panic("this should not happen")
			}

			virtualBatch := gorgonia.Must(gorgonia.Slice(x, gorgonia.S(start, end)))

			ret, err := nn.BN(BNOpts{
				Momentum:  opts.Momentum,
				Epsilon:   opts.Epsilon,
				Inferring: opts.Inferring,
				ScaleInit: opts.ScaleInit,
				BiasInit:  opts.BiasInit,
			})(virtualBatch)
			if err != nil {
				return nil, err
			}

			nodes = append(nodes, ret)
		}

		ret, err := (gorgonia.Concat(0, nodes...))
		if err != nil {
			return nil, fmt.Errorf("error concatenating %d nodes: %w", len(nodes), err)
		}

		return gorgonia.Reshape(ret, xShape)
	}
}
