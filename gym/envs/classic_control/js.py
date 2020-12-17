__all__ = ['js']

# Don't look below, you will not understand this Python code :) I don't.

from js2py.pyjs import *
# setting scope
var = Scope( JS_BUILTINS )
set_global_object(var)

# Code follows:
var.registers(['Vec2D', 'rk4'])
@Js
def PyJs_anonymous_0_(this, arguments, var=var):
    var = Scope({'this':this, 'arguments':arguments}, var)
    var.registers(['def', 'create'])
    @Js
    def PyJs_anonymous_1_(x, y, this, arguments, var=var):
        var = Scope({'x':x, 'y':y, 'this':this, 'arguments':arguments}, var)
        var.registers(['obj', 'y', 'x'])
        var.put('obj', var.get('Object').callprop('create', var.get('def')))
        var.get('obj').callprop('setXY', var.get('x'), var.get('y'))
        return var.get('obj')
    PyJs_anonymous_1_._set_name('anonymous')
    var.put('create', PyJs_anonymous_1_)
    @Js
    def PyJs_anonymous_2_(this, arguments, var=var):
        var = Scope({'this':this, 'arguments':arguments}, var)
        var.registers([])
        return var.get(u"this").get('_x')
    PyJs_anonymous_2_._set_name('anonymous')
    @Js
    def PyJs_anonymous_3_(value, this, arguments, var=var):
        var = Scope({'value':value, 'this':this, 'arguments':arguments}, var)
        var.registers(['value'])
        var.get(u"this").put('_x', var.get('value'))
    PyJs_anonymous_3_._set_name('anonymous')
    @Js
    def PyJs_anonymous_4_(this, arguments, var=var):
        var = Scope({'this':this, 'arguments':arguments}, var)
        var.registers([])
        return var.get(u"this").get('_y')
    PyJs_anonymous_4_._set_name('anonymous')
    @Js
    def PyJs_anonymous_5_(value, this, arguments, var=var):
        var = Scope({'value':value, 'this':this, 'arguments':arguments}, var)
        var.registers(['value'])
        var.get(u"this").put('_y', var.get('value'))
    PyJs_anonymous_5_._set_name('anonymous')
    @Js
    def PyJs_anonymous_6_(x, y, this, arguments, var=var):
        var = Scope({'x':x, 'y':y, 'this':this, 'arguments':arguments}, var)
        var.registers(['y', 'x'])
        var.get(u"this").put('_x', var.get('x'))
        var.get(u"this").put('_y', var.get('y'))
    PyJs_anonymous_6_._set_name('anonymous')
    @Js
    def PyJs_anonymous_7_(this, arguments, var=var):
        var = Scope({'this':this, 'arguments':arguments}, var)
        var.registers([])
        return var.get('Math').callprop('sqrt', ((var.get(u"this").get('_x')*var.get(u"this").get('_x'))+(var.get(u"this").get('_y')*var.get(u"this").get('_y'))))
    PyJs_anonymous_7_._set_name('anonymous')
    @Js
    def PyJs_anonymous_8_(length, this, arguments, var=var):
        var = Scope({'length':length, 'this':this, 'arguments':arguments}, var)
        var.registers(['length', 'angle'])
        var.put('angle', var.get(u"this").callprop('getAngle'))
        var.get(u"this").put('_x', (var.get('Math').callprop('cos', var.get('angle'))*var.get('length')))
        var.get(u"this").put('_y', (var.get('Math').callprop('sin', var.get('angle'))*var.get('length')))
    PyJs_anonymous_8_._set_name('anonymous')
    @Js
    def PyJs_anonymous_9_(this, arguments, var=var):
        var = Scope({'this':this, 'arguments':arguments}, var)
        var.registers([])
        return var.get('Math').callprop('atan2', var.get(u"this").get('_y'), var.get(u"this").get('_x'))
    PyJs_anonymous_9_._set_name('anonymous')
    @Js
    def PyJs_anonymous_10_(angle, this, arguments, var=var):
        var = Scope({'angle':angle, 'this':this, 'arguments':arguments}, var)
        var.registers(['length', 'angle'])
        var.put('length', var.get(u"this").callprop('getLength'))
        var.get(u"this").put('_x', (var.get('Math').callprop('cos', var.get('angle'))*var.get('length')))
        var.get(u"this").put('_y', (var.get('Math').callprop('sin', var.get('angle'))*var.get('length')))
    PyJs_anonymous_10_._set_name('anonymous')
    @Js
    def PyJs_anonymous_11_(vector, this, arguments, var=var):
        var = Scope({'vector':vector, 'this':this, 'arguments':arguments}, var)
        var.registers(['vector'])
        var.get(u"this").put('_x', var.get('vector').callprop('getX'), '+')
        var.get(u"this").put('_y', var.get('vector').callprop('getY'), '+')
    PyJs_anonymous_11_._set_name('anonymous')
    @Js
    def PyJs_anonymous_12_(vector, this, arguments, var=var):
        var = Scope({'vector':vector, 'this':this, 'arguments':arguments}, var)
        var.registers(['vector'])
        var.get(u"this").put('_x', var.get('vector').callprop('getX'), '-')
        var.get(u"this").put('_y', var.get('vector').callprop('getY'), '-')
    PyJs_anonymous_12_._set_name('anonymous')
    @Js
    def PyJs_anonymous_13_(value, this, arguments, var=var):
        var = Scope({'value':value, 'this':this, 'arguments':arguments}, var)
        var.registers(['value'])
        var.get(u"this").put('_x', var.get('value'), '*')
        var.get(u"this").put('_y', var.get('value'), '*')
    PyJs_anonymous_13_._set_name('anonymous')
    @Js
    def PyJs_anonymous_14_(value, this, arguments, var=var):
        var = Scope({'value':value, 'this':this, 'arguments':arguments}, var)
        var.registers(['value'])
        var.get(u"this").put('_x', var.get('value'), '/')
        var.get(u"this").put('_y', var.get('value'), '/')
    PyJs_anonymous_14_._set_name('anonymous')
    var.put('def', Js({'_x':Js(1.0),'_y':Js(0.0),'getX':PyJs_anonymous_2_,'setX':PyJs_anonymous_3_,'getY':PyJs_anonymous_4_,'setY':PyJs_anonymous_5_,'setXY':PyJs_anonymous_6_,'getLength':PyJs_anonymous_7_,'setLength':PyJs_anonymous_8_,'getAngle':PyJs_anonymous_9_,'setAngle':PyJs_anonymous_10_,'add':PyJs_anonymous_11_,'sub':PyJs_anonymous_12_,'mul':PyJs_anonymous_13_,'div':PyJs_anonymous_14_}))
    return Js({'create':var.get('create')})
PyJs_anonymous_0_._set_name('anonymous')
var.put('Vec2D', PyJs_anonymous_0_())
@Js
def PyJs_anonymous_15_(posX, posY, thrustX, thrustY, damping, velX, velY, this, arguments, var=var):
    var = Scope({'posX':posX, 'posY':posY, 'thrustX':thrustX, 'thrustY':thrustY, 'damping':damping, 'velX':velX, 'velY':velY, 'this':this, 'arguments':arguments}, var)
    var.registers(['newPosX', 'accel', 'dt', 'newVelX', 'pos', 'pos_kY', 'thrustX', 'newVelY', 'posVel', 'newVelVec', 'damping', 'velX', 'posY', 'vel', 'newPosVec', 'drag', 'newPosY', 'velY', 'pos_kX', 'vel_kX', 'thrust', 'posX', 'vel_kY', 'thrustY'])
    var.put('dt', Js(0.01774))
    var.put('thrust', var.get('Vec2D').callprop('create', var.get('thrustX'), var.get('thrustY')))
    var.put('accel', var.get('Vec2D').callprop('create', Js(0.0), Js(0.0)))
    var.put('vel', var.get('Vec2D').callprop('create', var.get('velX'), var.get('velY')))
    var.put('pos', var.get('Vec2D').callprop('create', var.get('posX'), var.get('posY')))
    var.put('drag', var.get('Vec2D').callprop('create', Js(0.0), Js(0.0)))
    var.get('drag').callprop('setLength', (var.get('vel').callprop('getLength')*var.get('damping')))
    var.get('drag').callprop('setAngle', var.get('vel').callprop('getAngle'))
    var.get('accel').callprop('add', var.get('thrust'))
    var.get('accel').callprop('sub', var.get('drag'))
    var.put('pos_kX', Js([]))
    var.put('pos_kY', Js([]))
    var.put('vel_kX', Js([]))
    var.put('vel_kY', Js([]))
    var.get('pos_kX').put('0', (var.get('dt')*var.get('vel').callprop('getX')))
    var.get('pos_kY').put('0', (var.get('dt')*var.get('vel').callprop('getY')))
    var.get('vel_kX').put('0', (var.get('dt')*var.get('accel').callprop('getX')))
    var.get('vel_kY').put('0', (var.get('dt')*var.get('accel').callprop('getY')))
    var.get('pos_kX').put('1', (var.get('dt')*(var.get('vel').callprop('getX')+(var.get('vel_kX').get('0')/Js(2.0)))))
    var.get('pos_kY').put('1', (var.get('dt')*(var.get('vel').callprop('getY')+(var.get('vel_kY').get('0')/Js(2.0)))))
    var.get('vel_kX').put('1', (var.get('dt')*var.get('accel').callprop('getX')))
    var.get('vel_kY').put('1', (var.get('dt')*var.get('accel').callprop('getY')))
    var.get('pos_kX').put('2', (var.get('dt')*(var.get('vel').callprop('getX')+(var.get('vel_kX').get('1')/Js(2.0)))))
    var.get('pos_kY').put('2', (var.get('dt')*(var.get('vel').callprop('getY')+(var.get('vel_kY').get('1')/Js(2.0)))))
    var.get('vel_kX').put('2', (var.get('dt')*var.get('accel').callprop('getX')))
    var.get('vel_kY').put('2', (var.get('dt')*var.get('accel').callprop('getY')))
    var.get('pos_kX').put('3', (var.get('dt')*(var.get('vel').callprop('getX')+var.get('vel_kX').get('2'))))
    var.get('pos_kY').put('3', (var.get('dt')*(var.get('vel').callprop('getY')+var.get('vel_kY').get('2'))))
    var.get('vel_kX').put('3', (var.get('dt')*var.get('accel').callprop('getX')))
    var.get('vel_kY').put('3', (var.get('dt')*var.get('accel').callprop('getY')))
    var.put('newPosX', ((Js(1.0)/Js(6.0))*(((var.get('pos_kX').get('0')+(Js(2.0)*var.get('pos_kX').get('1')))+(Js(2.0)*var.get('pos_kX').get('2')))+var.get('pos_kX').get('3'))))
    var.put('newPosY', ((Js(1.0)/Js(6.0))*(((var.get('pos_kY').get('0')+(Js(2.0)*var.get('pos_kY').get('1')))+(Js(2.0)*var.get('pos_kY').get('2')))+var.get('pos_kY').get('3'))))
    var.put('newVelX', ((Js(1.0)/Js(6.0))*(((var.get('vel_kX').get('0')+(Js(2.0)*var.get('vel_kX').get('1')))+(Js(2.0)*var.get('vel_kX').get('2')))+var.get('vel_kX').get('3'))))
    var.put('newVelY', ((Js(1.0)/Js(6.0))*(((var.get('vel_kY').get('0')+(Js(2.0)*var.get('vel_kY').get('1')))+(Js(2.0)*var.get('vel_kY').get('2')))+var.get('vel_kY').get('3'))))
    var.put('newPosVec', var.get('Vec2D').callprop('create', var.get('newPosX'), var.get('newPosY')))
    var.put('newVelVec', var.get('Vec2D').callprop('create', var.get('newVelX'), var.get('newVelY')))
    var.get('vel').callprop('add', var.get('newVelVec'))
    var.get('pos').callprop('add', var.get('newPosVec'))
    var.put('posVel', Js([var.get('pos').callprop('getX'), var.get('pos').callprop('getY'), var.get('vel').callprop('getX'), var.get('vel').callprop('getY')]))
    return var.get('posVel')
PyJs_anonymous_15_._set_name('anonymous')
var.put('rk4', PyJs_anonymous_15_)


# Add lib to the module scope
js = var.to_python()