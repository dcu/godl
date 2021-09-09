package godl

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func InBatches(x tensor.Tensor, batchSize int, cb func(v tensor.Tensor)) {
	totalSize := x.Shape()[0]
	batches := totalSize / batchSize

	for b := 0; b < batches; b++ {
		start := b * batchSize
		end := start + batchSize

		if start >= totalSize {
			break
		}

		if end > totalSize {
			end = totalSize
		}

		sliced, err := x.Slice(gorgonia.S(start, end))
		if err != nil {
			panic(err)
		}

		cb(sliced)
	}
}
