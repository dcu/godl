package godl

import (
	"fmt"
	"sync"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	layersCount = map[string]uint32{}
	countMutex  = sync.Mutex{}
)

type LayerType string

type Result struct {
	Output *gorgonia.Node
	Loss   *gorgonia.Node
	Nodes  gorgonia.Nodes
}

func (r Result) Shape() tensor.Shape {
	return r.Output.Shape()
}

func (r Result) Value() gorgonia.Value {
	return r.Output.Value()
}

// Layer defines a layer on the network
type Layer func(inputs ...*gorgonia.Node) (result Result, err error)

func AddLayer(typ string) LayerType {
	countMutex.Lock()
	layersCount[typ]++
	countMutex.Unlock()

	return LayerType(fmt.Sprintf("%s_%d", typ, layersCount[typ]))
}
