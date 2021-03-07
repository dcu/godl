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
	Drop      bool
}

func (o *DataLoaderOpts) setDefaults() {
	if o.BatchSize <= 0 {
		panic("batch size must be greater than 0")
	}
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

	if !opts.Drop {
		missingRows := opts.BatchSize - (numExamples % opts.BatchSize)

		rowsX := make([]tensor.Tensor, missingRows)
		rowsY := make([]tensor.Tensor, missingRows)

		for i := 0; i < missingRows; i++ {
			row := rand.Intn(numExamples)

			xS, err := x.Slice(gorgonia.S(row))
			if err != nil {
				panic(err)
			}

			newXShape := append(tensor.Shape{1}, x.Shape()[1:]...)
			err = xS.Reshape(newXShape...)
			if err != nil {
				panic(err)
			}

			rowsX[i] = xS

			yS, err := y.Slice(gorgonia.S(row))
			if err != nil {
				panic(err)
			}

			newYShape := append(tensor.Shape{1}, y.Shape()[1:]...)
			err = yS.Reshape(newYShape...)
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
	}

	batches := numExamples / opts.BatchSize

	dl := &DataLoader{
		x:             x,
		y:             y,
		opts:          opts,
		Rows:          numExamples,
		Batches:       batches,
		FeaturesShape: tensor.Shape(x.Shape()[1:]).Clone(),
	}

	dl.Reset()

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

func (dl *DataLoader) toMatrix(t tensor.Tensor) tensor.Shape {
	prevShape := t.Shape().Clone()

	err := t.Reshape(append(tensor.Shape{prevShape[0]}, tensor.Shape(prevShape[1:]).TotalSize())...)
	if err != nil {
		panic(err)
	}

	return prevShape
}

// Shuffle shuffles the data
func (dl *DataLoader) Shuffle() error {
	oldXShape := dl.toMatrix(dl.x)
	defer func() {
		_ = dl.x.Reshape(oldXShape...)
	}()

	iterX, err := native.MatrixF32(dl.x.(*tensor.Dense))
	if err != nil {
		return err
	}

	oldYShape := dl.toMatrix(dl.y)
	defer func() {
		_ = dl.y.Reshape(oldYShape...)
	}()

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

	dl.CurrentBatch++

	return xVal, yVal
}
