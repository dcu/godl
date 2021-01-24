package tabnet

import (
	"fmt"
	"sync"

	"gorgonia.org/gorgonia"
)

var (
	layersCount = map[string]uint32{}
	countMutex  = sync.Mutex{}
)

type layerType string

// Layer defines a layer on the network
type Layer func(inputs ...*gorgonia.Node) (output *gorgonia.Node, loss *gorgonia.Node, err error)

func incLayer(typ string) layerType {
	countMutex.Lock()
	layersCount[typ]++
	countMutex.Unlock()

	return layerType(fmt.Sprintf("%s_%d", typ, layersCount[typ]))
}
