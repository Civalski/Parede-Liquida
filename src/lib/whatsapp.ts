const WHATSAPP_PHONE = '5519992149699';
const WHATSAPP_MESSAGE =
	'Olá, gostaria de saber mais sobre o papel de parede liquido';

export function getWhatsAppHref(): string {
	return `https://wa.me/${WHATSAPP_PHONE}?text=${encodeURIComponent(WHATSAPP_MESSAGE)}`;
}
