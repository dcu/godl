package tabnet

import (
	"fmt"
	"log"
	"math"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// GBN implements a Ghost Batch Normalization: https://arxiv.org/pdf/1705.08741.pdf
// momentum defaults to 0.01 if 0 is passed
// epsilon defaults to 1e-5 if 0 is passed
func (nn *Model) GBN(x *gorgonia.Node, opts GBNOpts) (*gorgonia.Node, error) {
	opts.setDefaults()

	if x.Dims() == 4 {
		b, c, h, w := x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
		x = gorgonia.Must(gorgonia.Reshape(x, tensor.Shape{b, c * h * w}))
	}

	xShape := x.Shape()
	batchSize, inputSize := xShape[0], xShape[1]

	if opts.VirtualBatchSize > inputSize {
		opts.VirtualBatchSize = inputSize
	}

	batches := int(math.Ceil(float64(inputSize) / float64(opts.VirtualBatchSize)))
	nodes := make([]*gorgonia.Node, 0, batches)

	log.Printf("batches: %d", batches)
	log.Printf("momentum: %f eps: %f", opts.Momentum, opts.Epsilon)

	for b := 0; b < batchSize; b++ {
		// Split the Matrix
		vector := gorgonia.Must(gorgonia.Slice(x, gorgonia.S(b)))

		// Split the vector in virtual batches
		for vb := 0; vb < batches; vb++ {
			start := vb * opts.VirtualBatchSize
			if start > inputSize {
				break
			}

			end := start + opts.VirtualBatchSize
			if end > inputSize {
				break // FIXME: support end = inputSize
			}

			virtualBatch := gorgonia.Must(gorgonia.Slice(vector, gorgonia.S(start, end)))

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
	}

	ret, err := (gorgonia.Concat(0, nodes...))
	if err != nil {
		return nil, fmt.Errorf("error concatenating %d nodes: %w", len(nodes), err)
	}

	return gorgonia.Reshape(ret, xShape)
}
