"""Contact-gated per-leg phase scheduler; no model-state or force writes."""
import numpy as np
from run_sim import SIDES


class SupportCoordinator:
    def __init__(self, config):
        self.config = config
        self.phase = None
        self.state = ['stance'] * 4
        self.timer = np.zeros(4)
        self.events = []
        self.wait_seconds = np.zeros(4)
        self.wait_elapsed = np.zeros(4)
        self.contact_seconds = np.zeros(4)
        self.cursor = 0

    def event(self, t, i, name):
        self.events.append(dict(time_s=float(t), leg=SIDES[i], event=name))

    def advance(self, nominal, contact, dt, t):
        c = self.config
        gate = c.get('swing_gate_phase', .7)
        landing = c.get('landing_phase', .12)
        if self.phase is None:
            self.phase = nominal.copy()
            # Finish inherited swing phases before issuing new swing permissions.
            self.state = ['swing' if p >= gate else 'stance' for p in self.phase]
            return self.phase.copy()
        active = sum(s in ('swing', 'landing') for s in self.state)
        order = [0, 2, 1, 3]
        if c.get('support_recovery'):
            order = order[self.cursor:] + order[:self.cursor]
        else:
            order = list(range(4))
        for i in order:
            self.timer[i] += dt
            self.contact_seconds[i] = self.contact_seconds[i] + dt if contact[i] else 0.
            state = self.state[i]
            if state == 'landing':
                if contact[i] and (self.contact_seconds[i] if c.get('support_recovery') else self.timer[i]) >= c.get('support_confirm_s', .04):
                    self.state[i] = 'stance'; self.timer[i] = 0
                    active -= 1; self.event(t, i, 'support_confirmed')
                elif self.timer[i] >= c.get('landing_timeout_s', .2):
                    # A timeout is a failed landing confirmation, not a touchdown.
                    self.state[i] = 'stance'; self.timer[i] = 0
                    active -= 1; self.event(t, i, 'landing_timeout')
                continue
            if state == 'recover':
                target = c.get('recovery_phase', .45)
                self.phase[i] = max(target, self.phase[i] - c['frequency_hz'] * dt * .5)
                if self.phase[i] <= target:
                    self.state[i] = 'stance'; self.timer[i] = 0
                    self.event(t, i, 'recovery_finished')
                continue
            rate = 1. if state == 'swing' else c.get('support_phase_rate', .7)
            delta = c['frequency_hz'] * dt * rate
            old = self.phase[i]
            nxt = old + delta
            if state == 'stance' and old < gate <= nxt:
                others = contact.copy(); others[i] = False
                opposite = others[2:].any() if i < 2 else others[:2].any()
                allowed = (others.sum() >= c.get('minimum_support_feet', 2)
                    and opposite and active < c.get('maximum_swing_feet', 1))
                if not allowed:
                    self.phase[i] = gate - 1e-8
                    self.wait_seconds[i] += dt
                    self.wait_elapsed[i] += dt
                    if c.get('support_recovery') and self.wait_elapsed[i] >= c.get('support_wait_timeout_s', .12):
                        self.state[i] = 'recover'; self.timer[i] = 0; self.wait_elapsed[i] = 0
                        self.event(t, i, 'support_recovery')
                    continue
                self.state[i] = 'swing'; self.timer[i] = 0; self.wait_elapsed[i] = 0
                if c.get('support_recovery'):
                    self.cursor = ([0, 2, 1, 3].index(i) + 1) % 4
                active += 1; self.event(t, i, 'swing_granted')
            if state == 'swing' and old < landing <= nxt:
                self.phase[i] = landing
                self.state[i] = 'landing'; self.timer[i] = 0
                self.event(t, i, 'landing_wait')
            else:
                self.phase[i] = nxt % 1.
        return self.phase.copy()

    def diagnostics(self):
        return dict(events=self.events, waiting_seconds=dict(zip(SIDES,self.wait_seconds.tolist())),
                    final_states=self.state, final_phase=self.phase.tolist() if self.phase is not None else None)


class CoupledSupportCoordinator:
    """Preserve relative leg phases; temporarily slow a shared gait clock.

    Unlike the per-leg scheduler this is not a hard support-count guarantee.
    A bounded timeout restores progression and logs unmet support explicitly.
    """
    def __init__(self, config):
        self.config=config;self.phase=None;self.state=['stance']*4
        self.events=[];self.clock=0.;self.hold=0.;self.cooldown=0.
        self.previous=np.zeros(4,dtype=bool);self.landing_until=0.;self.slow_seconds=0.

    def advance(self, nominal, contact, dt, t):
        c=self.config
        if self.phase is None:
            self.phase=nominal.copy();self.previous=contact.copy();return self.phase.copy()
        touchdown=contact & ~self.previous;self.previous=contact.copy()
        if touchdown.any():self.landing_until=t+c.get('coupled_landing_hold_s',.03)
        gate=c.get('swing_gate_phase',.7)
        approaching=(self.phase>=gate-.08)&(self.phase<gate)
        needs=[]
        for i in np.flatnonzero(approaching):
            others=contact.copy();others[i]=False
            opposite=others[2:].any() if i<2 else others[:2].any()
            if not opposite or others.sum()<c.get('minimum_support_feet',2):needs.append(int(i))
        reason='nominal';rate=1.
        if needs and t>=self.cooldown:
            if self.hold==0:self.events.append(dict(time_s=t,event='support_slowdown',legs=[SIDES[i] for i in needs]))
            self.hold+=dt;reason='support';rate=c.get('coupled_slow_rate',.5)
            if self.hold>=c.get('coupled_wait_timeout_s',.06):
                self.events.append(dict(time_s=t,event='support_timeout'))
                self.cooldown=t+.15;self.hold=0.
        else:self.hold=0.
        if t<self.landing_until:rate=min(rate,.6);reason='landing'
        if rate<1:self.slow_seconds+=dt
        self.clock+=c['frequency_hz']*dt*rate
        self.phase=(self.phase+c['frequency_hz']*dt*rate)%1.
        self.state=['landing' if reason=='landing' else 'coupled']*4
        return self.phase.copy()

    def diagnostics(self):
        return dict(mode='coupled_clock',events=self.events,slow_seconds=self.slow_seconds,
                    relative_phase_preserved=True,final_phase=self.phase.tolist() if self.phase is not None else None)
