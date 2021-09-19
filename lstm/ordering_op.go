package lstm

import (
	"fmt"
	"hash"
	"hash/fnv"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type OrderingOp struct {
}

// Arity returns 1
func (i OrderingOp) Arity() int { return 1 }

// Type returns a → a
func (i OrderingOp) Type() hm.Type { return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a')) }

// InferShape returns the output shape as a function of the inputs
func (i OrderingOp) InferShape(ds ...gorgonia.DimSizer) (tensor.Shape, error) {
	return ds[0].(tensor.Shape), nil
}

// Do executes the op
func (i OrderingOp) Do(vs ...gorgonia.Value) (gorgonia.Value, error) {
	x := vs[0].(*tensor.Dense)

	indicesA := make([]int, 0, x.Shape()[1])
	for i := x.Shape()[1] - 1; i >= 0; i-- {
		indicesA = append(indicesA, i)
	}

	indices := tensor.New(
		tensor.Of(tensor.Int),
		tensor.WithShape(len(indicesA)),
		tensor.WithBacking(indicesA),
	)

	reversed, err := tensor.ByIndices(x, indices, 0)
	if err != nil {
		return nil, err
	}

	return reversed, nil
}

// ReturnsPtr indicates if the Op will return a pointer (allowing possible inplace edits) or by value
// if it's false, the return value of the Op will be a copy of its input
func (i OrderingOp) ReturnsPtr() bool { return true }

// CallsExtern returns false.
func (i OrderingOp) CallsExtern() bool { return false }

// OverwritesInput is a method which states which input the output will be overwriting.
// This allows for some efficiency gains as the underlying arrays wouldn't have to be re-allocated.
// The method returns an int instead of a bool because potentially different operations may be allowed
// to overwrite certain inputs. For example, consider an operation to increment a value:
// the IncrementOp would be a unary operator, and assuming we would like to overwrite the input,
// the retVal of overwriteInput() will be 0 (inputs[0]).
// -1 is returned if overwriting of input is disallowed
func (i OrderingOp) OverwritesInput() int { return -1 }

func (i OrderingOp) WriteHash(h hash.Hash) { fmt.Fprintf(h, i.String()) }

func (i OrderingOp) Hashcode() uint32 {
	h := fnv.New32a()
	i.WriteHash(h)
	return h.Sum32()
}

func (i OrderingOp) String() string { return "BackwardsOp" }
