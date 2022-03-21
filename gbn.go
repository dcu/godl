package godl

import (
	"math"

	"gorgonia.org/gorgonia"
)

// GBNOpts contains config options for the ghost batch normalization
type GBNOpts struct {
	Momentum         float64
	Epsilon          float64
	VirtualBatchSize int
	OutputDimension  int

	ScaleInit, BiasInit gorgonia.InitWFn
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
}

// GBN implements a Ghost Batch Normalization: https://arxiv.org/pdf/1705.08741.pdf
// momentum defaults to 0.01 if 0 is passed
// epsilon defaults to 1e-5 if 0 is passed
func GBN(nn *Model, opts GBNOpts) Layer {
	opts.setDefaults()

	lt := AddLayer("GBN")

	MustBeGreatherThan(lt, "OutputDimesion", opts.OutputDimension, 0)

	bn := BatchNorm1d(nn, BatchNormOpts{
		Momentum:  opts.Momentum,
		Epsilon:   opts.Epsilon,
		ScaleInit: opts.ScaleInit,
		BiasInit:  opts.BiasInit,
		InputSize: opts.OutputDimension,
	})

	return func(inputs ...*gorgonia.Node) (Result, error) {
		if err := nn.CheckArity(lt, inputs, 1); err != nil {
			return Result{}, err
		}

		x := inputs[0]
		xShape := x.Shape()
		inputSize := xShape[0]

		if opts.VirtualBatchSize > inputSize {
			opts.VirtualBatchSize = inputSize
		}

		if inputSize%opts.VirtualBatchSize != 0 {
			panic(ErrorF(lt, "input size (%d) must be divisible by virtual batch size (%v)", inputSize, opts.VirtualBatchSize))
		}

		batches := int(math.Ceil(float64(inputSize) / float64(opts.VirtualBatchSize)))
		nodes := make([]*gorgonia.Node, 0, batches)

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

			result, err := bn(virtualBatch)
			if err != nil {
				return Result{}, err
			}

			nodes = append(nodes, result.Output)
		}

		ret, err := gorgonia.Concat(0, nodes...)
		if err != nil {
			return Result{}, ErrorF(lt, "error concatenating %d nodes: %w", len(nodes), err)
		}

		// nn.Watch("gbnIn", inputs[0])
		// nn.Watch("gbnOut", ret)

		return Result{Output: ret}, nil
	}
}
