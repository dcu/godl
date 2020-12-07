package tabnet

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// LayersCount return the number of layers
func (t *TabNet) LayersCount() int {
	return len(t.learnables)
}

func (t *TabNet) addWeights(shape tensor.Shape) *gorgonia.Node {
	name := fmt.Sprintf("layer_%d_%d", len(t.learnables), shape.TotalSize())

	init := gorgonia.WithInit(gorgonia.GlorotN(1.0))
	if val, ok := t.model[name]; ok {
		init = gorgonia.WithValue(val)

		log.Printf("Assigned %d with shape %v pre-trained weights to %v", val.Size(), val.Shape(), name)
	} else {
		log.Printf("Assigned random weights to %v", name)
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
