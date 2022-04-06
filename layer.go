package godl

import (
	"fmt"
	"sync"
)

var (
	layersCount = map[string]uint32{}
	countMutex  = sync.Mutex{}
)

type LayerType string

func AddLayer(typ string) LayerType {
	countMutex.Lock()
	layersCount[typ]++
	countMutex.Unlock()

	return LayerType(fmt.Sprintf("%s_%d", typ, layersCount[typ]))
}
