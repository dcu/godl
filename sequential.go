package tabnet

import "gorgonia.org/gorgonia"

// Sequential runs the given layers one after the other
func (m *Model) Sequential(layers ...Layer) Layer {
	return func(inputs ...*gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, error) {
		err := m.checkArity("Sequential", inputs, 1)
		if err != nil {
			return nil, nil, err
		}

		var loss *gorgonia.Node
		x := inputs[0]
		losses := make([]*gorgonia.Node, 0, len(layers))

		for _, layer := range layers {
			x, loss, err = layer(x)
			if err != nil {
				return nil, nil, err
			}

			if loss != nil {
				losses = append(losses, loss)
			}
		}

		if len(losses) == 0 {
			return x, nil, nil
		}

		totalLoss := gorgonia.Must(gorgonia.ReduceAdd(losses))

		return x, totalLoss, nil
	}
}
