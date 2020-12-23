package tabnet

import (
	"fmt"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// LearnablesCount return the number of learnables
func (t *Model) LearnablesCount() int {
	return len(t.learnables)
}

func (t *Model) addWeights(shape tensor.Shape, initFN gorgonia.InitWFn) *gorgonia.Node {
	return t.addLearnable("weight", shape, initFN)
}

func (t *Model) addBias(shape tensor.Shape, initFN gorgonia.InitWFn) *gorgonia.Node {
	return t.addLearnable("bias", shape, initFN)
}

func (t *Model) addLearnable(name string, shape tensor.Shape, initFN gorgonia.InitWFn) *gorgonia.Node {
	name = fmt.Sprintf("%s_%d_%d", name, len(t.learnables), shape.TotalSize())

	if initFN == nil {
		initFN = gorgonia.GlorotN(1.0)
	}

	init := gorgonia.WithInit(initFN)
	if val, ok := t.model[name]; ok {
		init = gorgonia.WithValue(val)

		// log.Printf("Assigned %d with shape %v pre-trained values to %v", val.Size(), val.Shape(), name)
	} else {
		// log.Printf("Assigned new values to %v", name)
	}

	var w *gorgonia.Node

	if shape.Dims() == 2 {
		w = gorgonia.NewMatrix(
			t.g,
			tensor.Float64,
			gorgonia.WithShape(shape...),
			gorgonia.WithName(name),
			init,
		)
	} else {
		w = gorgonia.NewTensor(
			t.g,
			tensor.Float64,
			shape.Dims(),
			gorgonia.WithShape(shape...),
			gorgonia.WithName(name),
			init,
		)
	}

	t.learnables = append(t.learnables, w)

	return w
}
