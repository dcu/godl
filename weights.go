package tabnet

import (
	"fmt"
	"sync/atomic"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var learnablesCount int64

// NewNodeOpts defines the options to create a node
// Value has priority if it's not defined then it uses the InitFN if it's not defined it uses Glorot/Xavier(1.0)
// If UniqueName is empty an automatic one will be assigned.
type NewNodeOpts struct {
	UniqueName string
	Value      gorgonia.Value
	InitFN     gorgonia.InitWFn
}

// LearnablesCount return the number of learnables
func (t *Model) LearnablesCount() int64 {
	return learnablesCount
}

func (t *Model) AddWeights(lt LayerType, shape tensor.Shape, opts NewNodeOpts) *gorgonia.Node {
	return t.AddLearnable(lt, "weight", shape, opts)
}

func (t *Model) AddBias(lt LayerType, shape tensor.Shape, opts NewNodeOpts) *gorgonia.Node {
	return t.AddLearnable(lt, "bias", shape, opts)
}

func (t *Model) AddLearnable(lt LayerType, typ string, shape tensor.Shape, opts NewNodeOpts) *gorgonia.Node {
	if opts.UniqueName == "" {
		opts.UniqueName = fmt.Sprintf("%s.%d.%s.%d.%d", lt, learnablesCount, typ, len(t.learnables), shape.TotalSize())
	}

	w := t.CreateNode(shape, opts)
	t.learnables = append(t.learnables, w)

	return w
}

func (t *Model) CreateNode(shape tensor.Shape, opts NewNodeOpts) *gorgonia.Node {
	atomic.AddInt64(&learnablesCount, 1)

	var init gorgonia.NodeConsOpt

	if opts.Value != nil {
		init = gorgonia.WithValue(opts.Value)
	} else if opts.InitFN != nil {
		init = gorgonia.WithInit(opts.InitFN)
	} else {
		init = gorgonia.WithInit(gorgonia.GlorotN(1.0))
	}

	if t.loader != nil {
		val, err := t.loader.Load(opts.UniqueName)
		if err == nil {
			init = gorgonia.WithValue(val)
		}
	}

	var w *gorgonia.Node

	if shape.Dims() == 2 {
		w = gorgonia.NewMatrix(
			t.g,
			tensor.Float32,
			gorgonia.WithShape(shape...),
			gorgonia.WithName(opts.UniqueName),
			init,
		)
	} else {
		w = gorgonia.NewTensor(
			t.g,
			tensor.Float32,
			shape.Dims(),
			gorgonia.WithShape(shape...),
			gorgonia.WithName(opts.UniqueName),
			init,
		)
	}

	return w
}
