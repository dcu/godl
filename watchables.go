package tabnet

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

// Watch watches the given node
func (m *Model) Watch(name string, node *gorgonia.Node) {
	var v gorgonia.Value

	name = fmt.Sprintf("%s <%s>", name, node.Name())
	pointer := &v

	gorgonia.Read(node, pointer)

	m.watchables[name] = pointer
}
