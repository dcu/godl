package godl

import (
	"fmt"
	"sync/atomic"

	"github.com/fatih/color"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var weightsCount int64

// NewWeightsOpts defines the options to create a node
// Value has priority if it's not defined then it uses the InitFN if it's not defined it uses Glorot/Xavier(1.0)
// If UniqueName is empty an automatic one will be assigned.
type NewWeightsOpts struct {
	UniqueName string
	Value      gorgonia.Value
	InitFN     gorgonia.InitWFn

	// Fixed indicates that the weights won't be learnable. By default the weights are learnable
	Fixed bool
}

// WeightsCount return the number of learnables
func (t *Model) WeightsCount() int64 {
	return weightsCount
}

func (t *Model) AddWeights(lt LayerType, shape tensor.Shape, opts NewWeightsOpts) *gorgonia.Node {
	return t.AddLearnable(lt, "weight", shape, opts)
}

func (t *Model) AddBias(lt LayerType, shape tensor.Shape, opts NewWeightsOpts) *gorgonia.Node {
	return t.AddLearnable(lt, "bias", shape, opts)
}

func (t *Model) AddLearnable(lt LayerType, typ string, shape tensor.Shape, opts NewWeightsOpts) *gorgonia.Node {
	if opts.UniqueName == "" {
		opts.UniqueName = fmt.Sprintf("%s.%d.%s.%d.%d", lt, weightsCount, typ, len(t.learnables), shape.TotalSize())
	}

	w := t.CreateWeightsNode(shape, opts)
	t.learnables = append(t.learnables, w)

	return w
}

func (t *Model) CreateWeightsNode(shape tensor.Shape, opts NewWeightsOpts) *gorgonia.Node {
	atomic.AddInt64(&weightsCount, 1)

	var init gorgonia.NodeConsOpt

	if opts.Value != nil {
		init = gorgonia.WithValue(opts.Value)
	} else if opts.InitFN != nil {
		init = gorgonia.WithInit(opts.InitFN)
	} else {
		init = gorgonia.WithInit(gorgonia.GlorotN(1.0))
	}

	val, err := t.Storage.TensorByName(opts.UniqueName)
	if err == nil {
		color.Green("Loaded weights %v %v from storage", shape, opts.UniqueName)
		init = gorgonia.WithValue(val)
	} else {
		color.Yellow("Assigned random weights to %v %v", shape, opts.UniqueName)
	}

	var w *gorgonia.Node

	if shape.Dims() == 2 {
		w = gorgonia.NewMatrix(
			t.trainGraph,
			tensor.Float32,
			gorgonia.WithShape(shape...),
			gorgonia.WithName(opts.UniqueName),
			init,
		)
	} else {
		w = gorgonia.NewTensor(
			t.trainGraph,
			tensor.Float32,
			shape.Dims(),
			gorgonia.WithShape(shape...),
			gorgonia.WithName(opts.UniqueName),
			init,
		)
	}

	return w
}
