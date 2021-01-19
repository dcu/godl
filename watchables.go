package tabnet

import (
	"fmt"

	"github.com/fatih/color"
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

func (m Model) PrintWatchables() {
	for name, w := range m.watchables {
		if w != nil {
			fmt.Printf("[w] %s: %v\n%v\n\n", color.GreenString(name), (*w).Shape(), *w)
		}
	}
}
