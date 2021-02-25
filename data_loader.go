package deepzen

import (
	"math/rand"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
)

type DataLoaderOpts struct {
	Shuffle   bool
	BatchSize int
}

func (o *DataLoaderOpts) setDefaults() {
	MustBeGreatherThan(LayerType("DataLoader"), "BatchSize", o.BatchSize, 0)
}

type DataLoader struct {
	x    tensor.Tensor
	y    tensor.Tensor
	opts DataLoaderOpts

	FeaturesShape tensor.Shape
	Rows          int
	Batches       int
	CurrentBatch  int
}

// NewDataLoader creates a data loader with the given data and options
func NewDataLoader(x tensor.Tensor, y tensor.Tensor, opts DataLoaderOpts) *DataLoader {
	opts.setDefaults()

	var err error

	numExamples := x.Shape()[0]
	missingRows := opts.BatchSize - (numExamples % opts.BatchSize)

	rowsX := make([]tensor.Tensor, missingRows)
	rowsY := make([]tensor.Tensor, missingRows)

	for i := 0; i < missingRows; i++ {
		row := rand.Intn(numExamples)

		xS, err := x.Slice(gorgonia.S(row))
		if err != nil {
			panic(err)
		}

		err = xS.Reshape(1, x.Shape()[1])
		if err != nil {
			panic(err)
		}

		rowsX[i] = xS

		yS, err := y.Slice(gorgonia.S(row))
		if err != nil {
			panic(err)
		}

		err = yS.Reshape(1, y.Shape()[1])
		if err != nil {
			panic(err)
		}

		rowsY[i] = yS
	}

	x, err = tensor.Concat(0, x, rowsX...)
	if err != nil {
		panic(err)
	}

	y, err = tensor.Concat(0, y, rowsY...)
	if err != nil {
		panic(err)
	}

	numExamples += missingRows
	batches := numExamples / opts.BatchSize

	dl := &DataLoader{
		x:             x,
		y:             y,
		opts:          opts,
		Rows:          numExamples,
		Batches:       batches,
		FeaturesShape: tensor.Shape(x.Shape()[1:]),
	}

	return dl
}

// HasNext returns true if there's more batches to fetch
func (dl DataLoader) HasNext() bool {
	start := (dl.CurrentBatch + 1) * dl.opts.BatchSize

	if start >= dl.Rows {
		return false
	}

	return true
}

// Reset resets the iterator
func (dl *DataLoader) Reset() {
	dl.CurrentBatch = 0

	if dl.opts.Shuffle {
		err := dl.Shuffle()
		if err != nil {
			panic(err)
		}
	}
}

func (dl *DataLoader) Shuffle() error {
	iterX, err := native.MatrixF32(dl.x.(*tensor.Dense))
	if err != nil {
		return err
	}

	iterY, err := native.MatrixF32(dl.y.(*tensor.Dense))
	if err != nil {
		return err
	}

	tmp := make([]float32, dl.FeaturesShape.TotalSize())
	rand.Shuffle(dl.Rows, func(i, j int) {
		copy(tmp, iterX[i])
		copy(iterX[i], iterX[j])
		copy(iterX[j], tmp)

		copy(tmp, iterY[i])
		copy(iterY[i], iterY[j])
		copy(iterY[j], tmp)
	})

	return nil
}

// Next returns the next batch
func (dl *DataLoader) Next() (tensor.Tensor, tensor.Tensor) {
	start := dl.CurrentBatch * dl.opts.BatchSize
	end := start + dl.opts.BatchSize

	if start >= dl.Rows {
		return nil, nil
	}

	if end > dl.Rows {
		end = dl.Rows
	}

	inputSize := end - start

	xVal, err := dl.x.Slice(gorgonia.S(start, end))
	if err != nil {
		panic(err)
	}

	yVal, err := dl.y.Slice(gorgonia.S(start, end))
	if err != nil {
		panic(err)
	}

	err = xVal.(*tensor.Dense).Reshape(append(tensor.Shape{inputSize}, dl.FeaturesShape...)...)
	if err != nil {
		panic(err)
	}

	dl.CurrentBatch = (dl.CurrentBatch + 1) % dl.Batches

	return xVal, yVal
}
