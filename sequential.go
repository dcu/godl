package tabnet

import "gorgonia.org/gorgonia"

// Sequential runs the given layers one after the other
func (m *Model) Sequential(layers ...Layer) Layer {
	return func(inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
		err := m.checkArity("Sequential", inputs, 1)
		if err != nil {
			return nil, err
		}

		x := inputs[0]

		for _, layer := range layers {
			x, err = layer(x)
			if err != nil {
				return nil, err
			}
		}

		return x, nil
	}
}
