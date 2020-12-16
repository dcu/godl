package tabnet

import "gorgonia.org/gorgonia"

func hadamardProd(a *gorgonia.Node, b *gorgonia.Node) (*gorgonia.Node, error) {
	if !a.IsMatrix() || !b.IsMatrix() {
		return gorgonia.HadamardProd(a, b)
	}

	aShape := a.Shape()
	bShape := b.Shape()

	if aShape.Eq(bShape) {
		return gorgonia.HadamardProd(a, b)
	}

	var leftPattern, rightPattern []byte

	if aShape[1] > bShape[1] {
		rightPattern = []byte{1}
	} else {
		leftPattern = []byte{1}
	}

	return gorgonia.BroadcastHadamardProd(a, b, leftPattern, rightPattern)
}
