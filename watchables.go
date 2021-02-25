package deepzen

import (
	"fmt"

	"github.com/fatih/color"
	"gorgonia.org/gorgonia"
)

type watchable struct {
	name string
	node *gorgonia.Value
}

// Watch watches the given node
func (m *Model) Watch(name string, node *gorgonia.Node) {
	var v gorgonia.Value

	name = fmt.Sprintf("%s <%s>", name, node.Name())
	pointer := &v

	gorgonia.Read(node, pointer)

	m.watchables = append(m.watchables, watchable{name, pointer})
}

func (m Model) PrintWatchables() {
	for _, w := range m.watchables {
		if w.node != nil {
			fmt.Printf("[w] %s: %v\n%v\n\n", color.GreenString(w.name), (*w.node).Shape(), *w.node)
		}
	}
}
