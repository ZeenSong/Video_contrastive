from betty.problems import ImplicitProblem
from betty.engine import Engine
from betty.configs import Config, EngineConfig

class Inner(ImplicitProblem):
    def __init__(self, name, config, inner_func, module=None, optimizer=None, scheduler=None, train_data_loader=None, extra_config=None):
        super().__init__(name, config, module, optimizer, scheduler, train_data_loader, extra_config)
        self.inner_func = inner_func
        self.args = extra_config
    
    def training_step(self, X):
        inner_loss = self.inner_func(self.module, X, self.args)
        return inner_loss

    def on_inner_loop_start(self):
        self.module.load_state_dict(self.outer.module.state_dict())

class Outer(ImplicitProblem):
    def __init__(self, name, config, outter_func, module=None, optimizer=None, scheduler=None, train_data_loader=None, extra_config=None):
        super().__init__(name, config, module, optimizer, scheduler, train_data_loader, extra_config)
        self.outter_func = outter_func
        self.args = extra_config
        
    def training_step(self, X):
        outter_loss = self.outter_func(self.module, X, self.args)
        return outter_loss

def create_engine(model, outer_optim, inner_optim, cfg, inner_func, outter_func, train_data_loader):
    outer_config = Config(
        type=cfg.blotype, log_step=cfg.log_step, retain_graph=cfg.retain_graph
    )
    inner_config = Config(type=cfg.blotype, unroll_steps=cfg.inner_steps)

    engine_config = EngineConfig(
        train_iters=cfg.total_iter
    )
    outer = Outer(name="outer", outter_func=outter_func, module=model, config=outer_config, optimizer=outer_optim, extra_config=cfg, train_data_loader=train_data_loader)
    inner = Inner(name="inner", inner_func=inner_func, module=model, config=inner_config, optimizer=inner_optim, extra_config=cfg, train_data_loader=train_data_loader)
    problems = [outer, inner]
    u2l = {outer: [inner]}
    l2u = {inner: [outer]}
    dependencies = {"u2l": u2l, "l2u": l2u}
    engine = Engine(
        config=engine_config, problems=problems, dependencies=dependencies
    )
    return engine