package godl

// Sequential runs the given layers one after the other
func Sequential(m *Model, modules ...Module) ModuleList {
	_ = AddLayer("Sequential")

	list := ModuleList{}
	list.Add(modules...)

	return list
}
