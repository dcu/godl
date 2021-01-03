package tabnet

import (
	"fmt"
	"hash"
	"hash/fnv"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type EmbeddingOpts struct {
	WeightsInit gorgonia.InitWFn
}

// Embedding implements a embedding layer
func (m *Model) Embedding(embeddingSize int, embeddingDim int, opts EmbeddingOpts) Layer {
	w := m.addWeights(tensor.Shape{embeddingSize, embeddingDim}, opts.WeightsInit)

	return func(inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
		err := m.checkArity("Embedding", inputs, 1)
		if err != nil {
			return nil, err
		}

		x := inputs[0]
		xShape := x.Shape()

		x = gorgonia.Must(gorgonia.Reshape(x, tensor.Shape{xShape.TotalSize(), 1}))

		nodes := make(gorgonia.Nodes, xShape.TotalSize())

		for i := 0; i < xShape.TotalSize(); i++ {
			index := gorgonia.Must(gorgonia.Slice(x, gorgonia.S(i)))
			oneHot := gorgonia.Must(oneHotAt(index, embeddingSize))

			nodes[i] = gorgonia.Must(gorgonia.Mul(oneHot, w))
		}

		result := gorgonia.Must(gorgonia.Concat(0, nodes...))

		r := gorgonia.Must(gorgonia.Reshape(result, append(xShape, embeddingDim)))

		return r, nil
	}
}

func buildOneHotAt(index int, classes int) tensor.Tensor {
	oneHotBacking := make([]float64, classes)
	oneHotBacking[int(index)] = 1.0

	return tensor.New(
		tensor.WithShape(1, classes),
		tensor.WithBacking(oneHotBacking),
	)
}

type oneHotOp struct {
	classes int
}

func newOneHotOp(classes int) *oneHotOp {
	oneHotOp := &oneHotOp{
		classes: classes,
	}

	return oneHotOp
}

func oneHotAt(index *gorgonia.Node, classes int) (*gorgonia.Node, error) {
	op := newOneHotOp(classes)

	return gorgonia.ApplyOp(op, index)
}

func (op *oneHotOp) Arity() int {
	return 1
}

func (op *oneHotOp) ReturnsPtr() bool { return false }

func (op *oneHotOp) CallsExtern() bool { return false }

func (op *oneHotOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "oneHotOp{}(%v)", op.classes)
}

func (op *oneHotOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op *oneHotOp) String() string {
	return fmt.Sprintf("oneHotOp{}(%v)", op.classes)
}

func (op *oneHotOp) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	return tensor.Shape{op.classes, 1}, nil
}

func (op *oneHotOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a)
}

func (op *oneHotOp) OverwritesInput() int { return -1 }

func (op *oneHotOp) checkInput(inputs ...gorgonia.Value) (gorgonia.Scalar, error) {
	if len(inputs) != op.Arity() {
		return nil, fmt.Errorf("wrong number of parameters for oneHotOp %d, expected %d", len(inputs), op.Arity())
	}

	index, ok := inputs[0].(gorgonia.Scalar)
	if !ok {
		return nil, errors.Errorf("Expected argument 1 to be a Tensor, got %T", inputs[0])
	}

	return index, nil
}

func (op *oneHotOp) Do(inputs ...gorgonia.Value) (gorgonia.Value, error) {
	index, err := op.checkInput(inputs...)
	if err != nil {
		return nil, err
	}

	i := index.Data().(float64)

	output := buildOneHotAt(int(i), op.classes)

	return output, nil
}

// ensure it complies with the Op interface
var (
	_ gorgonia.Op = &oneHotOp{}
)
