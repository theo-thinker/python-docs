<template>
  <div class="animated-logo" @mouseover="startAnimation" @mouseout="stopAnimation">
    <div class="logo-container" :style="{ transform: `rotate(${rotation}deg)` }">
      <img :src="logoUrl" alt="Python Logo" class="logo-image"/>
    </div>
  </div>
</template>

<script setup>
import {ref, onMounted, onUnmounted} from 'vue';
// 导入SVG图片
import logoUrl from '../../../public/logo.svg';

const rotation = ref(0);
let animationFrame = null;
let direction = 1;
let animating = false;

const animate = () => {
  if (!animating) return;

  rotation.value += 0.5 * direction;

  if (rotation.value > 10) direction = -1;
  if (rotation.value < -10) direction = 1;

  animationFrame = requestAnimationFrame(animate);
};

const startAnimation = () => {
  if (animating) return;
  animating = true;
  animate();
};

const stopAnimation = () => {
  animating = false;
  cancelAnimationFrame(animationFrame);

  // 平滑回到初始位置
  const resetAnimation = () => {
    if (Math.abs(rotation.value) < 0.5) {
      rotation.value = 0;
      return;
    }

    rotation.value = rotation.value * 0.9;
    animationFrame = requestAnimationFrame(resetAnimation);
  };

  resetAnimation();
};

onMounted(() => {
  // 初始动画
  setTimeout(() => {
    startAnimation();
    setTimeout(stopAnimation, 2000);
  }, 500);
});

onUnmounted(() => {
  cancelAnimationFrame(animationFrame);
});
</script>

<style scoped>
.animated-logo {
  display: inline-block;
  cursor: pointer;
  perspective: 1000px;
}

.logo-container {
  transition: transform 0.05s ease-out;
  transform-origin: center center;
}

.logo-image {
  width: 40px;
  height: 40px;
  transition: filter 0.3s ease;
}

.animated-logo:hover .logo-image {
  filter: drop-shadow(0 0 8px rgba(69, 132, 182, 0.6));
}
</style> 