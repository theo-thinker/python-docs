<template>
  <div 
    class="back-to-top" 
    :class="{ visible: isVisible }" 
    @click="scrollToTop"
  >
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="18 15 12 9 6 15"></polyline>
    </svg>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';

const isVisible = ref(false);
let scrollListener = null;

const checkScroll = () => {
  isVisible.value = window.pageYOffset > 300;
};

const scrollToTop = () => {
  window.scrollTo({
    top: 0,
    behavior: 'smooth'
  });
};

onMounted(() => {
  scrollListener = window.addEventListener('scroll', checkScroll);
  checkScroll(); // 初始检查
});

onUnmounted(() => {
  if (scrollListener) {
    window.removeEventListener('scroll', checkScroll);
  }
});
</script>

<style scoped>
.back-to-top {
  position: fixed;
  right: 20px;
  bottom: 20px;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background-color: var(--vp-c-brand);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transform: translateY(20px);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  cursor: pointer;
  z-index: 99;
}

.back-to-top.visible {
  opacity: 0.9;
  transform: translateY(0);
}

.back-to-top:hover {
  opacity: 1;
  transform: translateY(-3px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.back-to-top svg {
  width: 20px;
  height: 20px;
  color: white;
}

@media (max-width: 768px) {
  .back-to-top {
    right: 15px;
    bottom: 15px;
    width: 36px;
    height: 36px;
  }
  
  .back-to-top svg {
    width: 18px;
    height: 18px;
  }
}
</style> 