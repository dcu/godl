package tabnet

import (
	"fmt"
	"sync/atomic"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var learnablesCount int64

// LearnablesCount return the number of learnables
func (t *Model) LearnablesCount() int64 {
	return learnablesCount
}

func (t *Model) AddWeights(lt LayerType, shape tensor.Shape, initFN gorgonia.InitWFn) *gorgonia.Node {
	return t.AddLearnable(lt, "weight", shape, initFN)
}

func (t *Model) AddBias(lt LayerType, shape tensor.Shape, initFN gorgonia.InitWFn) *gorgonia.Node {
	return t.AddLearnable(lt, "bias", shape, initFN)
}

func (t *Model) AddLearnable(lt LayerType, name string, shape tensor.Shape, initFN gorgonia.InitWFn) *gorgonia.Node {
	atomic.AddInt64(&learnablesCount, 1)

	name = fmt.Sprintf("%s.%d.%s.%d.%d", lt, learnablesCount, name, len(t.learnables), shape.TotalSize())

	if initFN == nil {
		initFN = gorgonia.GlorotN(1.0)
	}

	init := gorgonia.WithInit(initFN)
	if val, ok := t.model[name]; ok {
		init = gorgonia.WithValue(val)
	}

	var w *gorgonia.Node

	if shape.Dims() == 2 {
		w = gorgonia.NewMatrix(
			t.g,
			tensor.Float32,
			gorgonia.WithShape(shape...),
			gorgonia.WithName(name),
			init,
		)
	} else {
		w = gorgonia.NewTensor(
			t.g,
			tensor.Float32,
			shape.Dims(),
			gorgonia.WithShape(shape...),
			gorgonia.WithName(name),
			init,
		)
	}

	t.learnables = append(t.learnables, w)

	return w
}
