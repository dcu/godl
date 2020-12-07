package tabnet

import (
	"log"
	"math"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// GBN implements a Ghost Batch Normalization: https://arxiv.org/pdf/1705.08741.pdf
// momentum defaults to 0.01 if 0 is passed
// epsilon defaults to 1e-5 if 0 is passed
func (t *TabNet) GBN(x *gorgonia.Node, virtualBatchSize int, momentum float64, epsilon float64) (*gorgonia.Node, error) {
	if momentum <= 0.0 {
		momentum = 0.01
	}

	if epsilon <= 0.0 {
		epsilon = 1e-5
	}

	xShape := x.Shape()
	batchSize, inputSize := xShape[0], xShape[1]
	batches := int(math.Floor(float64(inputSize) / float64(virtualBatchSize))) // FIXME: Use Ceil otherwise some inputs are ignored

	nodes := make([]*gorgonia.Node, 0, batches)

	for b := 0; b < batchSize; b++ {
		// Split the Matrix
		vector := gorgonia.Must(gorgonia.Slice(x, gorgonia.S(b)))

		log.Printf("%v", vector.Shape())

		// Split the vector in virtual batches
		for vb := 0; vb < batches; vb++ {
			start := vb * virtualBatchSize
			end := start + virtualBatchSize

			if end > inputSize {
				end = inputSize
			}

			virtualBatch := gorgonia.Must(gorgonia.Slice(vector, gorgonia.S(start, end)))
			virtualBatch = gorgonia.Must(gorgonia.Reshape(virtualBatch, tensor.Shape{1, virtualBatch.Shape().TotalSize(), 1, 1}))

			log.Printf("%v", virtualBatch.Shape())

			ret, _, _, _, err := gorgonia.BatchNorm(virtualBatch, nil, nil, momentum, epsilon)
			if err != nil {
				return nil, err
			}

			nodes = append(nodes, ret)
		}
	}

	ret := gorgonia.Must(gorgonia.Concat(0, nodes...))

	return gorgonia.Reshape(ret, xShape)
}
