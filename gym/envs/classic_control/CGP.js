var Vec2D = (function(){
  //exposed methods:
  var create = function(x, y){
    var obj = Object.create(def);
    obj.setXY(x, y);

    return obj;
  };

  //Vec2D definition:
  var def ={
    _x: 1,
    _y: 0,

    getX: function()
    {
      return this._x;
    },

    setX: function(value)
    {
      this._x = value;
    },

    getY: function()
    {
      return this._y;
    },

    setY: function(value)
    {
      this._y = value;
    },

    setXY: function(x, y)
    {
      this._x = x;
      this._y = y;
    },

    getLength: function()
    {
      return Math.sqrt(this._x * this._x + this._y * this._y);
    },

    setLength: function(length)
    {
      var angle = this.getAngle();
      this._x = Math.cos(angle) * length;
      this._y = Math.sin(angle) * length;
    },

    getAngle: function()
    {
      return Math.atan2(this._y, this._x);
    },

    setAngle: function(angle)
    {
      var length = this.getLength();
      this._x = Math.cos(angle) * length;
      this._y = Math.sin(angle) * length;
    },

    add: function(vector)
    {
      this._x += vector.getX();
      this._y += vector.getY();
    },

    sub: function(vector)
    {
      this._x -= vector.getX();
      this._y -= vector.getY();
    },

    mul: function(value)
    {
      this._x *= value;
      this._y *= value;
    },

    div: function(value)
    {
      this._x /= value;
      this._y /= value;
    }
  };

  return {create:create};
}());

/////////////////////////////////////////////////////////////////////////



    /**** Animal def update method ****/
var rk4 = function(posX, posY, thrustX, thrustY, damping, velX, velY){
    const dt = 0.01774;

    var thrust = Vec2D.create(thrustX, thrustY);
    var accel = Vec2D.create(0,0);
    var vel = Vec2D.create(velX, velY);
    var pos = Vec2D.create(posX, posY);
    var drag = Vec2D.create(0,0);

    drag.setLength(vel.getLength() * damping);
    drag.setAngle(vel.getAngle());
	  //This might break?
    accel.add(thrust);
    accel.sub(drag);

      //begin runge-kutte 4
    var pos_kX = [];
    var pos_kY = [];
    var vel_kX = [];
    var vel_kY = [];
    pos_kX[0] = dt*vel.getX();
    pos_kY[0] = dt*vel.getY();
    vel_kX[0] = dt*accel.getX();
    vel_kY[0] = dt*accel.getY();

    pos_kX[1] = dt*(vel.getX() + (vel_kX[0]/2));
    pos_kY[1] = dt*(vel.getY() + (vel_kY[0]/2));

      // For the  time being, force is either on or off, max or zero.
      // Will work to fix this in future versions. I've made it so that the RK4 step is still set up and just needs to be added too
    vel_kX[1] = dt*accel.getX();
    vel_kY[1] = dt*accel.getY();

    pos_kX[2] = dt*(vel.getX() + (vel_kX[1]/2));
    pos_kY[2] = dt*(vel.getY() + (vel_kY[1]/2));
    vel_kX[2] = dt*accel.getX();
    vel_kY[2] = dt*accel.getY();

    pos_kX[3] = dt*(vel.getX() + vel_kX[2]);
    pos_kY[3] = dt*(vel.getY() + vel_kY[2]);
    vel_kX[3] = dt*accel.getX();
    vel_kY[3] = dt*accel.getY();

    var newPosX = (1/6)*(pos_kX[0]+(2*pos_kX[1])+(2*pos_kX[2]) + pos_kX[3]);
    var newPosY = (1/6)*(pos_kY[0]+(2*pos_kY[1])+(2*pos_kY[2]) + pos_kY[3]);

    var newVelX = (1/6)*(vel_kX[0]+(2*vel_kX[1])+(2*vel_kX[2]) + vel_kX[3]);
    var newVelY = (1/6)*(vel_kY[0]+(2*vel_kY[1])+(2*vel_kY[2]) + vel_kY[3]);

    var newPosVec = Vec2D.create(newPosX, newPosY);
    var newVelVec = Vec2D.create(newVelX, newVelY);


    vel.add(newVelVec); //Adds acceleration due to applied force to velocity
    pos.add(newPosVec); //Adds velocity to positon;

    var posVel = [pos.getX(), pos.getY(), vel.getX(), vel.getY()]

    return posVel;
      /*
      this.vel.add(this.thrust); //Adds acceleration due to applied force to velocity
      this.vel.sub(this.drag); //Subtracts acceleration due to drag force from velocity
      this.pos.add(this.vel); //Adds velocity to positon;
      */
  } // END update method
