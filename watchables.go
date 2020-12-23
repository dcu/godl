package tabnet

import "gorgonia.org/gorgonia"

var (
	watchables = make([]interface{}, 0)
)

// Watch watches the given node
func Watch(nodes ...*gorgonia.Node) {
	for _, node := range nodes {
		watchables = append(watchables, node)
	}
}
