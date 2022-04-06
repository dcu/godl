package godl

import (
	"math"

	"gorgonia.org/gorgonia"
)

// GhostBatchNormOpts contains config options for the ghost batch normalization
type GhostBatchNormOpts struct {
	Momentum         float64
	Epsilon          float64
	VirtualBatchSize int
	OutputDimension  int

	ScaleInit, BiasInit gorgonia.InitWFn
}

func (o *GhostBatchNormOpts) setDefaults() {
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

type GhostBatchNormModule struct {
	model *Model
	layer LayerType
	bn    *BatchNormModule
	opts  GhostBatchNormOpts
}

func (m *GhostBatchNormModule) Forward(inputs ...*Node) Nodes {
	if err := m.model.CheckArity(m.layer, inputs, 1); err != nil {
		panic(err)
	}

	x := inputs[0]
	xShape := x.Shape()
	inputSize := xShape[0]

	if m.opts.VirtualBatchSize > inputSize {
		m.opts.VirtualBatchSize = inputSize
	}

	if inputSize%m.opts.VirtualBatchSize != 0 {
		panic(ErrorF(m.layer, "input size (%d) must be divisible by virtual batch size (%v)", inputSize, m.opts.VirtualBatchSize))
	}

	batches := int(math.Ceil(float64(inputSize) / float64(m.opts.VirtualBatchSize)))
	nodes := make([]*gorgonia.Node, 0, batches)

	// Split the vector in virtual batches
	for vb := 0; vb < batches; vb++ {
		start := vb * m.opts.VirtualBatchSize
		if start > inputSize {
			break
		}

		end := start + m.opts.VirtualBatchSize
		if end > inputSize {
			panic("this should not happen")
		}

		virtualBatch := gorgonia.Must(gorgonia.Slice(x, gorgonia.S(start, end)))

		result := m.bn.Forward(virtualBatch)

		nodes = append(nodes, result...)
	}

	ret, err := gorgonia.Concat(0, nodes...)
	if err != nil {
		panic(ErrorF(m.layer, "error concatenating %d nodes: %w", len(nodes), err))
	}

	return Nodes{ret}
}

// GhostBatchNorm implements a Ghost Batch Normalization: https://arxiv.org/pdf/1705.08741.pdf
// momentum defaults to 0.01 if 0 is passed
// epsilon defaults to 1e-5 if 0 is passed
func GhostBatchNorm(nn *Model, opts GhostBatchNormOpts) *GhostBatchNormModule {
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

	return &GhostBatchNormModule{
		model: nn,
		layer: lt,
		bn:    bn,
		opts:  opts,
	}
}
